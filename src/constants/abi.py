SONIC_SWAP_ABI = [
    {
    "anonymous": False,
    "inputs": [
      {
        "indexed": True,
        "internalType": "address",
        "name": "sender",
        "type": "address"
      },
      {
        "indexed": False,
        "internalType": "uint256",
        "name": "amount0In",
        "type": "uint256"
      },
      {
        "indexed": False,
        "internalType": "address",
        "name": "_tokenIn",
        "type": "address"
      },
      {
        "indexed": True,
        "internalType": "address",
        "name": "to",
        "type": "address"
      },
      {
        "indexed": False,
        "internalType": "bool",
        "name": "stable",
        "type": "bool"
      }
    ],
    "name": "Swap",
    "type": "event"
  }
]

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_from", "type": "address"},
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transferFrom",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "owner", "type": "address"},
            {"indexed": True, "name": "spender", "type": "address"},
            {"indexed": False, "name": "value", "type": "uint256"}
        ],
        "name": "Approval",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "from", "type": "address"},
            {"indexed": True, "name": "to", "type": "address"},
            {"indexed": False, "name": "value", "type": "uint256"}
        ],
        "name": "Transfer",
        "type": "event"
    }
]

SILO_ABI = [
    {
      "inputs": [
        {
          "internalType": "contract ISiloFactory",
          "name": "_siloFactory",
          "type": "address"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "constructor"
    },
    {
      "stateMutability": "payable",
      "type": "receive"
    },
    {
      "inputs": [],
      "name": "DOMAIN_SEPARATOR",
      "outputs": [
        {
          "internalType": "bytes32",
          "name": "",
          "type": "bytes32"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "accrueInterest",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "accruedInterest",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_interestRateModel",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "_daoFee",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "_deployerFee",
          "type": "uint256"
        }
      ],
      "name": "accrueInterestForConfig",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "owner",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "spender",
          "type": "address"
        }
      ],
      "name": "allowance",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "spender",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "value",
          "type": "uint256"
        }
      ],
      "name": "approve",
      "outputs": [
        {
          "internalType": "bool",
          "name": "result",
          "type": "bool"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "asset",
      "outputs": [
        {
          "internalType": "address",
          "name": "assetTokenAddress",
          "type": "address"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "account",
          "type": "address"
        }
      ],
      "name": "balanceOf",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_account",
          "type": "address"
        }
      ],
      "name": "balanceOfAndTotalSupply",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_assets",
          "type": "uint256"
        },
        {
          "internalType": "address",
          "name": "_receiver",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "borrow",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_assets",
          "type": "uint256"
        },
        {
          "internalType": "address",
          "name": "_receiver",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "borrowSameAsset",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_shares",
          "type": "uint256"
        },
        {
          "internalType": "address",
          "name": "_receiver",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "borrowShares",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_owner",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_spender",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "_amount",
          "type": "uint256"
        }
      ],
      "name": "burn",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_target",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "_value",
          "type": "uint256"
        },
        {
          "internalType": "enum ISilo.CallType",
          "name": "_callType",
          "type": "uint8"
        },
        {
          "internalType": "bytes",
          "name": "_input",
          "type": "bytes"
        }
      ],
      "name": "callOnBehalfOfSilo",
      "outputs": [
        {
          "internalType": "bool",
          "name": "success",
          "type": "bool"
        },
        {
          "internalType": "bytes",
          "name": "result",
          "type": "bytes"
        }
      ],
      "stateMutability": "payable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "config",
      "outputs": [
        {
          "internalType": "contract ISiloConfig",
          "name": "siloConfig",
          "type": "address"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_shares",
          "type": "uint256"
        }
      ],
      "name": "convertToAssets",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_shares",
          "type": "uint256"
        },
        {
          "internalType": "enum ISilo.AssetType",
          "name": "_assetType",
          "type": "uint8"
        }
      ],
      "name": "convertToAssets",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_assets",
          "type": "uint256"
        },
        {
          "internalType": "enum ISilo.AssetType",
          "name": "_assetType",
          "type": "uint8"
        }
      ],
      "name": "convertToShares",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_assets",
          "type": "uint256"
        }
      ],
      "name": "convertToShares",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "decimals",
      "outputs": [
        {
          "internalType": "uint8",
          "name": "",
          "type": "uint8"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_assets",
          "type": "uint256"
        },
        {
          "internalType": "address",
          "name": "_receiver",
          "type": "address"
        }
      ],
      "name": "deposit",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_assets",
          "type": "uint256"
        },
        {
          "internalType": "address",
          "name": "_receiver",
          "type": "address"
        },
        {
          "internalType": "enum ISilo.CollateralType",
          "name": "_collateralType",
          "type": "uint8"
        }
      ],
      "name": "deposit",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "eip712Domain",
      "outputs": [
        {
          "internalType": "bytes1",
          "name": "fields",
          "type": "bytes1"
        },
        {
          "internalType": "string",
          "name": "name",
          "type": "string"
        },
        {
          "internalType": "string",
          "name": "version",
          "type": "string"
        },
        {
          "internalType": "uint256",
          "name": "chainId",
          "type": "uint256"
        },
        {
          "internalType": "address",
          "name": "verifyingContract",
          "type": "address"
        },
        {
          "internalType": "bytes32",
          "name": "salt",
          "type": "bytes32"
        },
        {
          "internalType": "uint256[]",
          "name": "extensions",
          "type": "uint256[]"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "factory",
      "outputs": [
        {
          "internalType": "contract ISiloFactory",
          "name": "",
          "type": "address"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_token",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "_amount",
          "type": "uint256"
        }
      ],
      "name": "flashFee",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "fee",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "contract IERC3156FlashBorrower",
          "name": "_receiver",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_token",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "_amount",
          "type": "uint256"
        },
        {
          "internalType": "bytes",
          "name": "_data",
          "type": "bytes"
        }
      ],
      "name": "flashLoan",
      "outputs": [
        {
          "internalType": "bool",
          "name": "success",
          "type": "bool"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_from",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_to",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "_amount",
          "type": "uint256"
        }
      ],
      "name": "forwardTransferFromNoChecks",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "getCollateralAndDebtTotalsStorage",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "totalCollateralAssets",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "totalDebtAssets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "getCollateralAndProtectedTotalsStorage",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "totalCollateralAssets",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "totalProtectedAssets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "getCollateralAssets",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "totalCollateralAssets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "getDebtAssets",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "totalDebtAssets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "getLiquidity",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "liquidity",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "getSiloStorage",
      "outputs": [
        {
          "internalType": "uint192",
          "name": "daoAndDeployerRevenue",
          "type": "uint192"
        },
        {
          "internalType": "uint64",
          "name": "interestRateTimestamp",
          "type": "uint64"
        },
        {
          "internalType": "uint256",
          "name": "protectedAssets",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "collateralAssets",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "debtAssets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "enum ISilo.AssetType",
          "name": "_assetType",
          "type": "uint8"
        }
      ],
      "name": "getTotalAssetsStorage",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "totalAssetsByType",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "hookReceiver",
      "outputs": [
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "hookSetup",
      "outputs": [
        {
          "components": [
            {
              "internalType": "address",
              "name": "hookReceiver",
              "type": "address"
            },
            {
              "internalType": "uint24",
              "name": "hooksBefore",
              "type": "uint24"
            },
            {
              "internalType": "uint24",
              "name": "hooksAfter",
              "type": "uint24"
            },
            {
              "internalType": "uint24",
              "name": "tokenType",
              "type": "uint24"
            }
          ],
          "internalType": "struct IShareToken.HookSetup",
          "name": "",
          "type": "tuple"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "contract ISiloConfig",
          "name": "_config",
          "type": "address"
        }
      ],
      "name": "initialize",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "isSolvent",
      "outputs": [
        {
          "internalType": "bool",
          "name": "",
          "type": "bool"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "maxBorrow",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "maxAssets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "maxBorrowSameAsset",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "maxAssets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "maxBorrowShares",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "maxShares",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        }
      ],
      "name": "maxDeposit",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "maxAssets",
          "type": "uint256"
        }
      ],
      "stateMutability": "pure",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_token",
          "type": "address"
        }
      ],
      "name": "maxFlashLoan",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "maxLoan",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        }
      ],
      "name": "maxMint",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "maxShares",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_owner",
          "type": "address"
        },
        {
          "internalType": "enum ISilo.CollateralType",
          "name": "_collateralType",
          "type": "uint8"
        }
      ],
      "name": "maxRedeem",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "maxShares",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_owner",
          "type": "address"
        }
      ],
      "name": "maxRedeem",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "maxShares",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "maxRepay",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "maxRepayShares",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_owner",
          "type": "address"
        },
        {
          "internalType": "enum ISilo.CollateralType",
          "name": "_collateralType",
          "type": "uint8"
        }
      ],
      "name": "maxWithdraw",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "maxAssets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_owner",
          "type": "address"
        }
      ],
      "name": "maxWithdraw",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "maxAssets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_shares",
          "type": "uint256"
        },
        {
          "internalType": "address",
          "name": "_receiver",
          "type": "address"
        }
      ],
      "name": "mint",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_shares",
          "type": "uint256"
        },
        {
          "internalType": "address",
          "name": "_receiver",
          "type": "address"
        },
        {
          "internalType": "enum ISilo.CollateralType",
          "name": "_collateralType",
          "type": "uint8"
        }
      ],
      "name": "mint",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_owner",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "_amount",
          "type": "uint256"
        }
      ],
      "name": "mint",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "name",
      "outputs": [
        {
          "internalType": "string",
          "name": "",
          "type": "string"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "owner",
          "type": "address"
        }
      ],
      "name": "nonces",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "owner",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "spender",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "value",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "deadline",
          "type": "uint256"
        },
        {
          "internalType": "uint8",
          "name": "v",
          "type": "uint8"
        },
        {
          "internalType": "bytes32",
          "name": "r",
          "type": "bytes32"
        },
        {
          "internalType": "bytes32",
          "name": "s",
          "type": "bytes32"
        }
      ],
      "name": "permit",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_assets",
          "type": "uint256"
        }
      ],
      "name": "previewBorrow",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_shares",
          "type": "uint256"
        }
      ],
      "name": "previewBorrowShares",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_assets",
          "type": "uint256"
        },
        {
          "internalType": "enum ISilo.CollateralType",
          "name": "_collateralType",
          "type": "uint8"
        }
      ],
      "name": "previewDeposit",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_assets",
          "type": "uint256"
        }
      ],
      "name": "previewDeposit",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_shares",
          "type": "uint256"
        },
        {
          "internalType": "enum ISilo.CollateralType",
          "name": "_collateralType",
          "type": "uint8"
        }
      ],
      "name": "previewMint",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_shares",
          "type": "uint256"
        }
      ],
      "name": "previewMint",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_shares",
          "type": "uint256"
        }
      ],
      "name": "previewRedeem",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_shares",
          "type": "uint256"
        },
        {
          "internalType": "enum ISilo.CollateralType",
          "name": "_collateralType",
          "type": "uint8"
        }
      ],
      "name": "previewRedeem",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_assets",
          "type": "uint256"
        }
      ],
      "name": "previewRepay",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_shares",
          "type": "uint256"
        }
      ],
      "name": "previewRepayShares",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_assets",
          "type": "uint256"
        }
      ],
      "name": "previewWithdraw",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_assets",
          "type": "uint256"
        },
        {
          "internalType": "enum ISilo.CollateralType",
          "name": "_collateralType",
          "type": "uint8"
        }
      ],
      "name": "previewWithdraw",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_shares",
          "type": "uint256"
        },
        {
          "internalType": "address",
          "name": "_receiver",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_owner",
          "type": "address"
        }
      ],
      "name": "redeem",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_shares",
          "type": "uint256"
        },
        {
          "internalType": "address",
          "name": "_receiver",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_owner",
          "type": "address"
        },
        {
          "internalType": "enum ISilo.CollateralType",
          "name": "_collateralType",
          "type": "uint8"
        }
      ],
      "name": "redeem",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_assets",
          "type": "uint256"
        },
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "repay",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_shares",
          "type": "uint256"
        },
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "repayShares",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "silo",
      "outputs": [
        {
          "internalType": "contract ISilo",
          "name": "",
          "type": "address"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "siloConfig",
      "outputs": [
        {
          "internalType": "contract ISiloConfig",
          "name": "",
          "type": "address"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "switchCollateralToThisSilo",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "symbol",
      "outputs": [
        {
          "internalType": "string",
          "name": "",
          "type": "string"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint24",
          "name": "_hooksBefore",
          "type": "uint24"
        },
        {
          "internalType": "uint24",
          "name": "_hooksAfter",
          "type": "uint24"
        }
      ],
      "name": "synchronizeHooks",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "totalAssets",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "totalManagedAssets",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "totalSupply",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_to",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "_amount",
          "type": "uint256"
        }
      ],
      "name": "transfer",
      "outputs": [
        {
          "internalType": "bool",
          "name": "result",
          "type": "bool"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_from",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_to",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "_amount",
          "type": "uint256"
        }
      ],
      "name": "transferFrom",
      "outputs": [
        {
          "internalType": "bool",
          "name": "result",
          "type": "bool"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_shares",
          "type": "uint256"
        },
        {
          "internalType": "address",
          "name": "_owner",
          "type": "address"
        },
        {
          "internalType": "enum ISilo.CollateralType",
          "name": "_transitionFrom",
          "type": "uint8"
        }
      ],
      "name": "transitionCollateral",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "updateHooks",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "utilizationData",
      "outputs": [
        {
          "components": [
            {
              "internalType": "uint256",
              "name": "collateralAssets",
              "type": "uint256"
            },
            {
              "internalType": "uint256",
              "name": "debtAssets",
              "type": "uint256"
            },
            {
              "internalType": "uint64",
              "name": "interestRateTimestamp",
              "type": "uint64"
            }
          ],
          "internalType": "struct ISilo.UtilizationData",
          "name": "",
          "type": "tuple"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_assets",
          "type": "uint256"
        },
        {
          "internalType": "address",
          "name": "_receiver",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_owner",
          "type": "address"
        }
      ],
      "name": "withdraw",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_assets",
          "type": "uint256"
        },
        {
          "internalType": "address",
          "name": "_receiver",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_owner",
          "type": "address"
        },
        {
          "internalType": "enum ISilo.CollateralType",
          "name": "_collateralType",
          "type": "uint8"
        }
      ],
      "name": "withdraw",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "withdrawFees",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "hooksBefore",
          "type": "uint256"
        }
      ],
      "name": "AccruedInterest",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": True,
          "internalType": "address",
          "name": "owner",
          "type": "address"
        },
        {
          "indexed": True,
          "internalType": "address",
          "name": "spender",
          "type": "address"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "value",
          "type": "uint256"
        }
      ],
      "name": "Approval",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": True,
          "internalType": "address",
          "name": "sender",
          "type": "address"
        },
        {
          "indexed": True,
          "internalType": "address",
          "name": "receiver",
          "type": "address"
        },
        {
          "indexed": True,
          "internalType": "address",
          "name": "owner",
          "type": "address"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "name": "Borrow",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": True,
          "internalType": "address",
          "name": "borrower",
          "type": "address"
        }
      ],
      "name": "CollateralTypeChanged",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": True,
          "internalType": "address",
          "name": "sender",
          "type": "address"
        },
        {
          "indexed": True,
          "internalType": "address",
          "name": "owner",
          "type": "address"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "name": "Deposit",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": True,
          "internalType": "address",
          "name": "sender",
          "type": "address"
        },
        {
          "indexed": True,
          "internalType": "address",
          "name": "owner",
          "type": "address"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "name": "DepositProtected",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [],
      "name": "EIP712DomainChanged",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "amount",
          "type": "uint256"
        }
      ],
      "name": "FlashLoan",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": False,
          "internalType": "uint24",
          "name": "hooksBefore",
          "type": "uint24"
        },
        {
          "indexed": False,
          "internalType": "uint24",
          "name": "hooksAfter",
          "type": "uint24"
        }
      ],
      "name": "HooksUpdated",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": False,
          "internalType": "uint64",
          "name": "version",
          "type": "uint64"
        }
      ],
      "name": "Initialized",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": True,
          "internalType": "address",
          "name": "notificationReceiver",
          "type": "address"
        },
        {
          "indexed": False,
          "internalType": "bool",
          "name": "success",
          "type": "bool"
        }
      ],
      "name": "NotificationSent",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": True,
          "internalType": "address",
          "name": "sender",
          "type": "address"
        },
        {
          "indexed": True,
          "internalType": "address",
          "name": "owner",
          "type": "address"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "name": "Repay",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": True,
          "internalType": "address",
          "name": "from",
          "type": "address"
        },
        {
          "indexed": True,
          "internalType": "address",
          "name": "to",
          "type": "address"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "value",
          "type": "uint256"
        }
      ],
      "name": "Transfer",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": True,
          "internalType": "address",
          "name": "sender",
          "type": "address"
        },
        {
          "indexed": True,
          "internalType": "address",
          "name": "receiver",
          "type": "address"
        },
        {
          "indexed": True,
          "internalType": "address",
          "name": "owner",
          "type": "address"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "name": "Withdraw",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": True,
          "internalType": "address",
          "name": "sender",
          "type": "address"
        },
        {
          "indexed": True,
          "internalType": "address",
          "name": "receiver",
          "type": "address"
        },
        {
          "indexed": True,
          "internalType": "address",
          "name": "owner",
          "type": "address"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "assets",
          "type": "uint256"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "shares",
          "type": "uint256"
        }
      ],
      "name": "WithdrawProtected",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "daoFees",
          "type": "uint256"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "deployerFees",
          "type": "uint256"
        }
      ],
      "name": "WithdrawnFeed",
      "type": "event"
    },
    {
      "inputs": [],
      "name": "AboveMaxLtv",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "AmountExceedsAllowance",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "BorrowNotPossible",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "CollateralSiloAlreadySet",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "CrossReentrantCall",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "ECDSAInvalidSignature",
      "type": "error"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "length",
          "type": "uint256"
        }
      ],
      "name": "ECDSAInvalidSignatureLength",
      "type": "error"
    },
    {
      "inputs": [
        {
          "internalType": "bytes32",
          "name": "s",
          "type": "bytes32"
        }
      ],
      "name": "ECDSAInvalidSignatureS",
      "type": "error"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "spender",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "allowance",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "needed",
          "type": "uint256"
        }
      ],
      "name": "ERC20InsufficientAllowance",
      "type": "error"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "sender",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "balance",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "needed",
          "type": "uint256"
        }
      ],
      "name": "ERC20InsufficientBalance",
      "type": "error"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "approver",
          "type": "address"
        }
      ],
      "name": "ERC20InvalidApprover",
      "type": "error"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "receiver",
          "type": "address"
        }
      ],
      "name": "ERC20InvalidReceiver",
      "type": "error"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "sender",
          "type": "address"
        }
      ],
      "name": "ERC20InvalidSender",
      "type": "error"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "spender",
          "type": "address"
        }
      ],
      "name": "ERC20InvalidSpender",
      "type": "error"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "deadline",
          "type": "uint256"
        }
      ],
      "name": "ERC2612ExpiredSignature",
      "type": "error"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "signer",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "owner",
          "type": "address"
        }
      ],
      "name": "ERC2612InvalidSigner",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "EarnedZero",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "FlashloanAmountTooBig",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "FlashloanFailed",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "InputCanBeAssetsOrShares",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "InputZeroShares",
      "type": "error"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "account",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "currentNonce",
          "type": "uint256"
        }
      ],
      "name": "InvalidAccountNonce",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "InvalidInitialization",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "NoLiquidity",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "NotEnoughLiquidity",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "NotInitializing",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "NotSolvent",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "NothingToWithdraw",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "OnlyHookReceiver",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "OnlySilo",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "OnlySiloConfig",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "OwnerIsZero",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "RecipientIsZero",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "RecipientNotSolventAfterTransfer",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "RepayTooHigh",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "ReturnZeroAssets",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "ReturnZeroShares",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "SenderNotSolventAfterTransfer",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "SiloInitialized",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "UnsupportedFlashloanToken",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "ZeroAmount",
      "type": "error"
    },
    {
      "inputs": [],
      "name": "ZeroTransfer",
      "type": "error"
    }
]

SILO_CONFIG = [
  {
    "type": "constructor",
    "inputs": [
      {
        "name": "_siloId",
        "type": "uint256",
        "internalType": "uint256"
      },
      {
        "name": "_configData0",
        "type": "tuple",
        "internalType": "struct ISiloConfig.ConfigData",
        "components": [
          {
            "name": "daoFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "deployerFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "silo",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "token",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "protectedShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "collateralShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "debtShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "solvencyOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtvOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "interestRateModel",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "lt",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationTargetLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "flashloanFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "hookReceiver",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "callBeforeQuote",
            "type": "bool",
            "internalType": "bool"
          }
        ]
      },
      {
        "name": "_configData1",
        "type": "tuple",
        "internalType": "struct ISiloConfig.ConfigData",
        "components": [
          {
            "name": "daoFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "deployerFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "silo",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "token",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "protectedShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "collateralShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "debtShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "solvencyOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtvOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "interestRateModel",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "lt",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationTargetLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "flashloanFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "hookReceiver",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "callBeforeQuote",
            "type": "bool",
            "internalType": "bool"
          }
        ]
      }
    ],
    "stateMutability": "nonpayable"
  },
  {
    "type": "function",
    "name": "SILO_ID",
    "inputs": [],
    "outputs": [
      {
        "name": "",
        "type": "uint256",
        "internalType": "uint256"
      }
    ],
    "stateMutability": "view"
  },
  {
    "type": "function",
    "name": "accrueInterestForBothSilos",
    "inputs": [],
    "outputs": [],
    "stateMutability": "nonpayable"
  },
  {
    "type": "function",
    "name": "accrueInterestForSilo",
    "inputs": [
      {
        "name": "_silo",
        "type": "address",
        "internalType": "address"
      }
    ],
    "outputs": [],
    "stateMutability": "nonpayable"
  },
  {
    "type": "function",
    "name": "borrowerCollateralSilo",
    "inputs": [
      {
        "name": "borrower",
        "type": "address",
        "internalType": "address"
      }
    ],
    "outputs": [
      {
        "name": "collateralSilo",
        "type": "address",
        "internalType": "address"
      }
    ],
    "stateMutability": "view"
  },
  {
    "type": "function",
    "name": "getAssetForSilo",
    "inputs": [
      {
        "name": "_silo",
        "type": "address",
        "internalType": "address"
      }
    ],
    "outputs": [
      {
        "name": "asset",
        "type": "address",
        "internalType": "address"
      }
    ],
    "stateMutability": "view"
  },
  {
    "type": "function",
    "name": "getCollateralShareTokenAndAsset",
    "inputs": [
      {
        "name": "_silo",
        "type": "address",
        "internalType": "address"
      },
      {
        "name": "_collateralType",
        "type": "uint8",
        "internalType": "enum ISilo.CollateralType"
      }
    ],
    "outputs": [
      {
        "name": "shareToken",
        "type": "address",
        "internalType": "address"
      },
      {
        "name": "asset",
        "type": "address",
        "internalType": "address"
      }
    ],
    "stateMutability": "view"
  },
  {
    "type": "function",
    "name": "getConfig",
    "inputs": [
      {
        "name": "_silo",
        "type": "address",
        "internalType": "address"
      }
    ],
    "outputs": [
      {
        "name": "config",
        "type": "tuple",
        "internalType": "struct ISiloConfig.ConfigData",
        "components": [
          {
            "name": "daoFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "deployerFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "silo",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "token",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "protectedShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "collateralShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "debtShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "solvencyOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtvOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "interestRateModel",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "lt",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationTargetLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "flashloanFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "hookReceiver",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "callBeforeQuote",
            "type": "bool",
            "internalType": "bool"
          }
        ]
      }
    ],
    "stateMutability": "view"
  },
  {
    "type": "function",
    "name": "getConfigsForBorrow",
    "inputs": [
      {
        "name": "_debtSilo",
        "type": "address",
        "internalType": "address"
      }
    ],
    "outputs": [
      {
        "name": "collateralConfig",
        "type": "tuple",
        "internalType": "struct ISiloConfig.ConfigData",
        "components": [
          {
            "name": "daoFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "deployerFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "silo",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "token",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "protectedShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "collateralShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "debtShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "solvencyOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtvOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "interestRateModel",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "lt",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationTargetLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "flashloanFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "hookReceiver",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "callBeforeQuote",
            "type": "bool",
            "internalType": "bool"
          }
        ]
      },
      {
        "name": "debtConfig",
        "type": "tuple",
        "internalType": "struct ISiloConfig.ConfigData",
        "components": [
          {
            "name": "daoFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "deployerFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "silo",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "token",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "protectedShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "collateralShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "debtShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "solvencyOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtvOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "interestRateModel",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "lt",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationTargetLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "flashloanFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "hookReceiver",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "callBeforeQuote",
            "type": "bool",
            "internalType": "bool"
          }
        ]
      }
    ],
    "stateMutability": "view"
  },
  {
    "type": "function",
    "name": "getConfigsForSolvency",
    "inputs": [
      {
        "name": "_borrower",
        "type": "address",
        "internalType": "address"
      }
    ],
    "outputs": [
      {
        "name": "collateralConfig",
        "type": "tuple",
        "internalType": "struct ISiloConfig.ConfigData",
        "components": [
          {
            "name": "daoFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "deployerFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "silo",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "token",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "protectedShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "collateralShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "debtShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "solvencyOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtvOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "interestRateModel",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "lt",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationTargetLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "flashloanFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "hookReceiver",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "callBeforeQuote",
            "type": "bool",
            "internalType": "bool"
          }
        ]
      },
      {
        "name": "debtConfig",
        "type": "tuple",
        "internalType": "struct ISiloConfig.ConfigData",
        "components": [
          {
            "name": "daoFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "deployerFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "silo",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "token",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "protectedShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "collateralShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "debtShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "solvencyOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtvOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "interestRateModel",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "lt",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationTargetLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "flashloanFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "hookReceiver",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "callBeforeQuote",
            "type": "bool",
            "internalType": "bool"
          }
        ]
      }
    ],
    "stateMutability": "view"
  },
  {
    "type": "function",
    "name": "getConfigsForWithdraw",
    "inputs": [
      {
        "name": "_silo",
        "type": "address",
        "internalType": "address"
      },
      {
        "name": "_depositOwner",
        "type": "address",
        "internalType": "address"
      }
    ],
    "outputs": [
      {
        "name": "depositConfig",
        "type": "tuple",
        "internalType": "struct ISiloConfig.DepositConfig",
        "components": [
          {
            "name": "silo",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "token",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "collateralShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "protectedShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "daoFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "deployerFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "interestRateModel",
            "type": "address",
            "internalType": "address"
          }
        ]
      },
      {
        "name": "collateralConfig",
        "type": "tuple",
        "internalType": "struct ISiloConfig.ConfigData",
        "components": [
          {
            "name": "daoFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "deployerFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "silo",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "token",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "protectedShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "collateralShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "debtShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "solvencyOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtvOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "interestRateModel",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "lt",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationTargetLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "flashloanFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "hookReceiver",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "callBeforeQuote",
            "type": "bool",
            "internalType": "bool"
          }
        ]
      },
      {
        "name": "debtConfig",
        "type": "tuple",
        "internalType": "struct ISiloConfig.ConfigData",
        "components": [
          {
            "name": "daoFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "deployerFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "silo",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "token",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "protectedShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "collateralShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "debtShareToken",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "solvencyOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtvOracle",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "interestRateModel",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "maxLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "lt",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationTargetLtv",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "liquidationFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "flashloanFee",
            "type": "uint256",
            "internalType": "uint256"
          },
          {
            "name": "hookReceiver",
            "type": "address",
            "internalType": "address"
          },
          {
            "name": "callBeforeQuote",
            "type": "bool",
            "internalType": "bool"
          }
        ]
      }
    ],
    "stateMutability": "view"
  },
  {
    "type": "function",
    "name": "getDebtShareTokenAndAsset",
    "inputs": [
      {
        "name": "_silo",
        "type": "address",
        "internalType": "address"
      }
    ],
    "outputs": [
      {
        "name": "shareToken",
        "type": "address",
        "internalType": "address"
      },
      {
        "name": "asset",
        "type": "address",
        "internalType": "address"
      }
    ],
    "stateMutability": "view"
  },
  {
    "type": "function",
    "name": "getDebtSilo",
    "inputs": [
      {
        "name": "_borrower",
        "type": "address",
        "internalType": "address"
      }
    ],
    "outputs": [
      {
        "name": "debtSilo",
        "type": "address",
        "internalType": "address"
      }
    ],
    "stateMutability": "view"
  },
  {
    "type": "function",
    "name": "getFeesWithAsset",
    "inputs": [
      {
        "name": "_silo",
        "type": "address",
        "internalType": "address"
      }
    ],
    "outputs": [
      {
        "name": "daoFee",
        "type": "uint256",
        "internalType": "uint256"
      },
      {
        "name": "deployerFee",
        "type": "uint256",
        "internalType": "uint256"
      },
      {
        "name": "flashloanFee",
        "type": "uint256",
        "internalType": "uint256"
      },
      {
        "name": "asset",
        "type": "address",
        "internalType": "address"
      }
    ],
    "stateMutability": "view"
  },
  {
    "type": "function",
    "name": "getShareTokens",
    "inputs": [
      {
        "name": "_silo",
        "type": "address",
        "internalType": "address"
      }
    ],
    "outputs": [
      {
        "name": "protectedShareToken",
        "type": "address",
        "internalType": "address"
      },
      {
        "name": "collateralShareToken",
        "type": "address",
        "internalType": "address"
      },
      {
        "name": "debtShareToken",
        "type": "address",
        "internalType": "address"
      }
    ],
    "stateMutability": "view"
  },
  {
    "type": "function",
    "name": "getSilos",
    "inputs": [],
    "outputs": [
      {
        "name": "silo0",
        "type": "address",
        "internalType": "address"
      },
      {
        "name": "silo1",
        "type": "address",
        "internalType": "address"
      }
    ],
    "stateMutability": "view"
  },
  {
    "type": "function",
    "name": "hasDebtInOtherSilo",
    "inputs": [
      {
        "name": "_thisSilo",
        "type": "address",
        "internalType": "address"
      },
      {
        "name": "_borrower",
        "type": "address",
        "internalType": "address"
      }
    ],
    "outputs": [
      {
        "name": "hasDebt",
        "type": "bool",
        "internalType": "bool"
      }
    ],
    "stateMutability": "view"
  },
  {
    "type": "function",
    "name": "onDebtTransfer",
    "inputs": [
      {
        "name": "_sender",
        "type": "address",
        "internalType": "address"
      },
      {
        "name": "_recipient",
        "type": "address",
        "internalType": "address"
      }
    ],
    "outputs": [],
    "stateMutability": "nonpayable"
  },
  {
    "type": "function",
    "name": "reentrancyGuardEntered",
    "inputs": [],
    "outputs": [
      {
        "name": "entered",
        "type": "bool",
        "internalType": "bool"
      }
    ],
    "stateMutability": "view"
  },
  {
    "type": "function",
    "name": "setOtherSiloAsCollateralSilo",
    "inputs": [
      {
        "name": "_borrower",
        "type": "address",
        "internalType": "address"
      }
    ],
    "outputs": [],
    "stateMutability": "nonpayable"
  },
  {
    "type": "function",
    "name": "setThisSiloAsCollateralSilo",
    "inputs": [
      {
        "name": "_borrower",
        "type": "address",
        "internalType": "address"
      }
    ],
    "outputs": [],
    "stateMutability": "nonpayable"
  },
  {
    "type": "function",
    "name": "turnOffReentrancyProtection",
    "inputs": [],
    "outputs": [],
    "stateMutability": "nonpayable"
  },
  {
    "type": "function",
    "name": "turnOnReentrancyProtection",
    "inputs": [],
    "outputs": [],
    "stateMutability": "nonpayable"
  },
  {
    "type": "error",
    "name": "CrossReentrancyNotActive",
    "inputs": []
  },
  {
    "type": "error",
    "name": "CrossReentrantCall",
    "inputs": []
  },
  {
    "type": "error",
    "name": "DebtExistInOtherSilo",
    "inputs": []
  },
  {
    "type": "error",
    "name": "FeeTooHigh",
    "inputs": []
  },
  {
    "type": "error",
    "name": "OnlyDebtShareToken",
    "inputs": []
  },
  {
    "type": "error",
    "name": "OnlySilo",
    "inputs": []
  },
  {
    "type": "error",
    "name": "OnlySiloOrTokenOrHookReceiver",
    "inputs": []
  },
  {
    "type": "error",
    "name": "WrongSilo",
    "inputs": []
  }
]

SILO_LENS_ABI = [
    {
      "inputs": [
        {
          "internalType": "contract ISilo",
          "name": "_silo",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "collateralBalanceOfUnderlying",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "borrowerCollateral",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "contract ISilo",
          "name": "_silo",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "collateralBalanceOfUnderlying",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "borrowerCollateral",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "contract ISilo",
          "name": "_silo",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "debtBalanceOfUnderlying",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "contract ISilo",
          "name": "_silo",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "debtBalanceOfUnderlying",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "borrowerDebt",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "contract ISilo",
          "name": "_silo",
          "type": "address"
        }
      ],
      "name": "getBorrowAPR",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "borrowAPR",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "contract ISilo",
          "name": "_silo",
          "type": "address"
        }
      ],
      "name": "getDepositAPR",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "depositAPR",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "contract ISilo",
          "name": "_silo",
          "type": "address"
        }
      ],
      "name": "getFeesAndFeeReceivers",
      "outputs": [
        {
          "internalType": "address",
          "name": "daoFeeReceiver",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "deployerFeeReceiver",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "daoFee",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "deployerFee",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "contract ISilo",
          "name": "_silo",
          "type": "address"
        }
      ],
      "name": "getInterestRateModel",
      "outputs": [
        {
          "internalType": "address",
          "name": "irm",
          "type": "address"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "contract ISilo",
          "name": "_silo",
          "type": "address"
        }
      ],
      "name": "getLt",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "lt",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "contract ISilo",
          "name": "_silo",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "getLtv",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "ltv",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "contract ISilo",
          "name": "_silo",
          "type": "address"
        }
      ],
      "name": "getMaxLtv",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "maxLtv",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "contract ISilo",
          "name": "_silo",
          "type": "address"
        }
      ],
      "name": "getRawLiquidity",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "liquidity",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "contract ISilo",
          "name": "_silo",
          "type": "address"
        },
        {
          "internalType": "contract IPartialLiquidation",
          "name": "_hook",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_borrower",
          "type": "address"
        }
      ],
      "name": "maxLiquidation",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "collateralToLiquidate",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "debtToRepay",
          "type": "uint256"
        },
        {
          "internalType": "bool",
          "name": "sTokenRequired",
          "type": "bool"
        },
        {
          "internalType": "bool",
          "name": "fullLiquidation",
          "type": "bool"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    }
]
  