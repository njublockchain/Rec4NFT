# %%
import clickhouse_connect
from web3 import Web3
import logging
from pprint import pprint

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s"
)

w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))

client = clickhouse_connect.get_client(
    host="localhost", username="default", password=""
)

sub_client = clickhouse_connect.get_client(
    host="localhost", username="default", password=""
)


from typing import Any, Dict, cast, Union

from eth_typing import HexStr
from eth_utils import event_abi_to_log_topic
from hexbytes import HexBytes
from web3._utils.abi import get_abi_input_names, get_abi_input_types, map_abi_data
from web3._utils.normalizers import BASE_RETURN_NORMALIZERS
from web3.contract import Contract
import web3

import eth_abi


class EventDecoder:
    def __init__(self, name, abi):
        self.name = name
        self.abi = [abi]

        self.contract = w3.eth.contract(abi=self.abi)

    def decode(self, log) -> Dict[str, Any]:
        event = getattr(self.contract.events, self.name)
        return event().process_log(log)['args']


OrderFulfilled_contract_abi = {
    "anonymous": False,
    "inputs": [
        {
            "indexed": False,
            "internalType": "bytes32",
            "name": "orderHash",
            "type": "bytes32",
        },
        {
            "indexed": True,
            "internalType": "address",
            "name": "offerer",
            "type": "address",
        },
        {"indexed": True, "internalType": "address", "name": "zone", "type": "address"},
        {
            "indexed": False,
            "internalType": "address",
            "name": "recipient",
            "type": "address",
        },
        {
            "components": [
                {"internalType": "enum ItemType", "name": "itemType", "type": "uint8"},
                {"internalType": "address", "name": "token", "type": "address"},
                {"internalType": "uint256", "name": "identifier", "type": "uint256"},
                {"internalType": "uint256", "name": "amount", "type": "uint256"},
            ],
            "indexed": False,
            "internalType": "struct SpentItem[]",
            "name": "offer",
            "type": "tuple[]",
        },
        {
            "components": [
                {"internalType": "enum ItemType", "name": "itemType", "type": "uint8"},
                {"internalType": "address", "name": "token", "type": "address"},
                {"internalType": "uint256", "name": "identifier", "type": "uint256"},
                {"internalType": "uint256", "name": "amount", "type": "uint256"},
                {
                    "internalType": "address payable",
                    "name": "recipient",
                    "type": "address",
                },
            ],
            "indexed": False,
            "internalType": "struct ReceivedItem[]",
            "name": "consideration",
            "type": "tuple[]",
        },
    ],
    "name": "OrderFulfilled",
    "type": "event",
}

OrderFulfilled_decoder = EventDecoder("OrderFulfilled", OrderFulfilled_contract_abi)

# %%

OrderFulfilled_stream = client.query_rows_stream(
    """
    select 
        *,
    from ethereum.events
    where
        blockNumber <= 18000000 AND
        (
            `address` = unhex('00000000006c3852cbEf3e08E8dF289169EdE581')
            OR `address` = unhex('00000000000006c7676171937C444f6BDe3D6282')
            OR `address` = unhex('0000000000000aD24e80fd803C6ac37206a45f15')
            OR `address` = unhex('00000000000001ad428e4906aE43D8F9852d0dD6') 
            OR `address` = unhex('00000000000000ADc04C56Bf30aC9d3c0aAF14dC')
        )
        AND arrayElement(`topics`, 1) = unhex('9d9af8e38d66c62e2c12f0225249fd9d721c54b83f48d9352c97c6cacdcb6f31')
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

column_names = OrderFulfilled_stream.source.column_names
count = 0

prev_transaction_hash = None
prev_match_log_index = None

from binascii import unhexlify

with OrderFulfilled_stream, open("seaport_0_18000000.txt", "w") as f:
    for row in OrderFulfilled_stream:
        data = row[column_names.index("data")]

        log = OrderFulfilled_decoder.decode({
            "address": row[column_names.index("address")],
            "topics": row[column_names.index("topics")], 
            "data": unhexlify(data),
            "logIndex": row[column_names.index("logIndex")],
            "transactionIndex": row[column_names.index("transactionIndex")],
            "transactionHash": row[column_names.index("transactionHash")],
            "blockHash": row[column_names.index("blockHash")],
            "blockNumber": row[column_names.index("blockNumber")],
        })
        # assert len(log['offer']) > 0 or len(log['consideration']) > 0, HexBytes(row[column_names.index("transactionHash")]).hex()

        log['orderHash'] = HexBytes(log['orderHash']).hex()

        f.write(
            json.dumps(
                {
                    "transactionHash": "0x"
                    + row[column_names.index("transactionHash")].hex(),
                    "blockNumber": row[column_names.index("blockNumber")],
                    "blockTimestamp": row[column_names.index("blockTimestamp")],
                    **log
                }
            ) + "\n"
        )
      
        # consideration = fee + payment + overflow

        count += 1
        if count % 1000 == 0:
            f.flush()
            logging.warning(f"count: {count}")

# %%
