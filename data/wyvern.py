# %%
import asyncio
import clickhouse_connect
from web3 import Web3
import logging

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
)

w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))

client = clickhouse_connect.get_client(
    host="localhost", username="default", password=""
)


from typing import Any, Dict

from eth_typing import HexStr
from web3._utils.normalizers import BASE_RETURN_NORMALIZERS


class EventDecoder:
    def __init__(self, name, abi):
        self.name = name
        self.abi = [abi]

        self.contract = w3.eth.contract(abi=self.abi)

    def decode(self, log) -> Dict[str, Any]:
        event = getattr(self.contract.events, self.name)
        return event().process_log(log)["args"]


OrdersMatched_contract_abi = {
    "anonymous": False,
    "inputs": [
        {"indexed": False, "name": "buyHash", "type": "bytes32"},
        {"indexed": False, "name": "sellHash", "type": "bytes32"},
        {"indexed": True, "name": "maker", "type": "address"},
        {"indexed": True, "name": "taker", "type": "address"},
        {"indexed": False, "name": "price", "type": "uint256"},
        {"indexed": True, "name": "metadata", "type": "bytes32"},
    ],
    "name": "OrdersMatched",
    "type": "event",
}


Transfer_contract_abi = {
    "anonymous": False,
    "inputs": [
        {"indexed": True, "name": "owner", "type": "address"},
        {"indexed": True, "name": "approved", "type": "address"},
        {"indexed": True, "name": "tokenId", "type": "uint256"},
    ],
    "name": "Approval",
    "type": "event",
}

OrdersMatched_decoder = EventDecoder("OrdersMatched", OrdersMatched_contract_abi)
Transfer_decoder = EventDecoder("Transfer", Transfer_contract_abi)

# %%
from binascii import unhexlify


def find_NFT_by_approval(
    transaction_hash, maker, match_log_index, prev_match_log_index=None
):
    # from maker to taker
    if type(transaction_hash) == bytes:
        transaction_hash = transaction_hash.hex()
    transaction_hash_no_prefix = transaction_hash.removeprefix("0x")
    maker_address_no_prefix = maker.removeprefix("0x").zfill(64)
    sql = f"""
        select 
            *,
            hex(`data`) as `data_hex`
        from ethereum.events
        where
            transactionHash = unhex('{transaction_hash_no_prefix}')
            AND arrayElement(`topics`, 1) = unhex('8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925')
            AND arrayElement(`topics`, 2) = unhex('{maker_address_no_prefix}')
            AND `logIndex` < {match_log_index}
    """
    if prev_match_log_index is not None:
        sql += f" AND `logIndex` > {prev_match_log_index}"
    sub_client = clickhouse_connect.get_client(
        host="localhost", username="default", password=""
    )
    results = sub_client.query(sql)
    column_names = results.column_names

    if len(results.result_rows) == 0:
        return None

    # assert len(results.result_rows) == 1, f"len(results.result_rows) = {len(results.result_rows)} on tx {transaction_hash}"

    nft_set = []

    for row in results.result_rows:
        data = row[column_names.index("data")]

        log = Transfer_decoder.decode(
            {
                "address": row[column_names.index("address")],
                "topics": row[column_names.index("topics")],
                "data": unhexlify(data),
                "logIndex": row[column_names.index("logIndex")],
                "transactionIndex": row[column_names.index("transactionIndex")],
                "transactionHash": row[column_names.index("transactionHash")],
                "blockHash": row[column_names.index("blockHash")],
                "blockNumber": row[column_names.index("blockNumber")],
            }
        )
        nft_set.append(
            {
                "collectionAddress": "0x" + row[column_names.index("address")].hex(),
                "tokenId": log["tokenId"],
            }
        )
    return nft_set


# find_NFT_by_approval("0xe31d22a0e8c81c10804931cc5c3c7bae9b15e89e0a70eeed722a8043347cd781", "0x07d01Ee0b61dc7D3D572A5Cb73F7Df6c05FEB1AA")

# %%

OrdersMatched_results = client.query(
    """
    select 
        *,
        hex(`data`) as `data_hex`
    from ethereum.events
    where
        blockNumber <= 18000000 AND
        (`address` = unhex('7be8076f4ea4a4ad08075c2508e481d6c946d12b')  OR `address` = unhex('7f268357a8c2552623316e2562d90e642bb538e5'))
        AND arrayElement(`topics`, 1) = unhex('c4109843e0b7d514e4c093114b863f8e7d8d9a458c372cd51bfe526b588006c9')
    ORDER BY transactionHash, logIndex
""",
    settings={
        "read_backoff_min_latency_ms": 0,
        "max_block_size": 100000,
        "connect_timeout": 100000,
        "send_timeout": 6000,
        "receive_timeout": 100000,
    },
)

import json

column_names = OrdersMatched_results.column_names

import threading

lock = threading.Lock()

f = open("wyvern_0_18000000_json.txt", "w")


async def handle_row(row, prev_match_log_index):
    data = row[column_names.index("data")]

    log = OrdersMatched_decoder.decode(
        {
            "address": row[column_names.index("address")],
            "topics": row[column_names.index("topics")],
            "data": unhexlify(data),
            "logIndex": row[column_names.index("logIndex")],
            "transactionIndex": row[column_names.index("transactionIndex")],
            "transactionHash": row[column_names.index("transactionHash")],
            "blockHash": row[column_names.index("blockHash")],
            "blockNumber": row[column_names.index("blockNumber")],
        }
    )

    nft_set = find_NFT_by_approval(
        row[column_names.index("transactionHash")],
        log["maker"],
        row[column_names.index("logIndex")],
        prev_match_log_index,
    )

    if nft_set is None:
        print(
            "0x"
            + row[column_names.index("transactionHash")].hex()
            + " is empty NFT set"
        )
        return

    lock.acquire()
    f.write(
        json.dumps(
            {
                "transactionHash": "0x"
                + row[column_names.index("transactionHash")].hex(),
                "blockNumber": row[column_names.index("blockNumber")],
                "blockTimestamp": row[column_names.index("blockTimestamp")],
                "marketplaceAddress": "0x" + row[column_names.index("address")].hex(),
                "NFTs": nft_set,
                "fromAddress": log["maker"],
                "toAddress": log["taker"],
                "price": log["price"],
            }
        )
        + "\n"
    )
    lock.release()


async def main():
    prev_transaction_hash = None
    prev_match_log_index = None

    count = 0
    awaits = set()

    for row in OrdersMatched_results.result_rows:
        if (
            prev_transaction_hash
            == "0x" + row[column_names.index("transactionHash")].hex()
        ):
            awaits.add(handle_row(row, prev_match_log_index))
        else:
            awaits.add(handle_row(row, None))

        prev_transaction_hash = "0x" + row[column_names.index("transactionHash")].hex()
        prev_match_log_index = row[column_names.index("logIndex")]

        count += 1
        if count % 1000 == 0:
            f.flush()
            logging.warning(f"count: {count}")

    await asyncio.gather(*awaits)
    f.close()


asyncio.run(main())


# %%
