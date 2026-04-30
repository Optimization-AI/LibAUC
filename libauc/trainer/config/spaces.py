class AUCMLossSpace:
    optimizer = {
        "type" : "PESG",
        "space" : {
            "lr" : {
                "val": (0.0001, 0.1),
                "default" : 0.1,
                "log": True
            },
            "epoch_decay" : {
                "val" : (0.0, 0.01),
                "default" : 0.002
            },
            "weight_decay" : {
                "val" : (0.0, 0.0002),
                "default" : 1e-5
            },
            "momentum" : {
                "val" : (0.8, 0.99),
                "default" : 0.9
            }
        }
    }
    loss = {
        "type" : "AUCMLoss",
        "space" : {
            "margin" : {
                "val" : [0.6, 0.8, 1.0],
                "default" : 1.0
            }
        }
    }

class MultiLabelAUCMLossSpace:
    optimizer = {
        "type" : "PESG",
        "space" : {
            "lr" : {
                "val": (0.0001, 0.1),
                "default" : 0.1,
                "log": True
            },
            "epoch_decay" : {
                "val" : (0.0, 0.01),
                "default" : 0.002
            },
            "weight_decay" : {
                "val" : (0.0, 0.0002),
                "default" : 1e-5
            },
            "momentum" : {
                "val" : (0.8, 0.99),
                "default" : 0.9
            }
        }
    }
    loss = {
        "type" : "MultiLabelAUCMLoss",
        "space" : {
            "margin" : {
                "val" : [0.6, 0.8, 1.0],
                "default" : 1.0
            }
        }
    }

class CompositionalAUCLossSpace:
    optimizer = {
        "type" : "PDSCA",
        "space" : {
            "lr" : {
                "val": (0.0001, 0.1),
                "default" : 0.1,
                "log": True
            },
            "epoch_decay" : {
                "val" : (0.0, 0.01),
                "default" : 0.002
            },
            "weight_decay" : {
                "val" : (0.0, 0.0002),
                "default" : 1e-5
            },
        }
    }
    loss = {
        "type" : "CompositionalAUCLoss",
        "space" : {
            "margin" : {
                "val" : [0.6, 0.8, 1.0],
                "default" : 1.0
            },
            "k" :{
                "val" : [1, 2, 4],
                "default" : 1
            }
        }
    }

class APLossSpace:
    optimizer = {
        "type" : "SOAP",
        "space" : {
            "lr" : {
                "val": (0.0001, 0.1),
                "default" : 0.001,
                "log": True
            },
            "momentum" : {
                "val" : (0.8, 0.99),
                "default" : 0.9
            },
            "weight_decay" : {
                "val" : (0.0, 0.0002),
                "default" : 1e-5
            },
        }
    }
    loss = {
        "type" : "APLoss",
        "space" : {
            "gamma" :{
                "val" : (0.0, 1.0),
                "default" : 0.9
            },
            "margin" : {
                "val" : [0.6, 0.8, 1.0],
                "default" : 1.0
            }
        }
    }

class mAPLossSpace:
    optimizer = {
        "type" : "SOAP",
        "space" : {
            "lr" : {
                "val": (0.0001, 0.1),
                "default" : 0.001,
                "log": True
            },
            "momentum" : {
                "val" : (0.8, 0.99),
                "default" : 0.9
            },
            "weight_decay" : {
                "val" : (0.0, 0.0002),
                "default" : 1e-5
            },
        }
    }
    loss = {
        "type" : "mAPLoss",
        "space" : {
            "gamma" :{
                "val" : (0.0, 1.0),
                "default" : 0.9
            },
            "margin" : {
                "val" : [0.6, 0.8, 1.0],
                "default" : 1.0
            }
        }
    }
    
class pAUC_CVaR_LossSpace:
    optimizer = {
        "type" : "SOPA",
        "space" : {
            "lr" : {
                "val": (0.0001, 0.1),
                "default" : 0.001,
                "log": True
            },
            "momentum" : {
                "val" : (0.8, 0.99),
                "default" : 0.9
            },
            "weight_decay" : {
                "val" : (0.0, 0.0002),
                "default" : 0
            }
        }
    }
    loss = {
        "type" : "pAUCLoss",
        "space" : {
            "mode" :{
                "val" : "SOPA"
            },
            "margin" : {
                "val" : [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "default" : 1.0
            },
            "beta" :{
                "val" : 0.2
            },
            "eta" : {
                "val" : (0.01, 10),
                "default" : 0.1,
                "log" : True
            }
        }
    }

class MultiLabelpAUC_CVaR_LossSpace:
    optimizer = {
        "type" : "SOPA",
        "space" : {
            "lr" : {
                "val": (0.0001, 0.1),
                "default" : 0.001,
                "log": True
            },
            "momentum" : {
                "val" : (0.8, 0.99),
                "default" : 0.9
            },
            "weight_decay" : {
                "val" : (0.0, 0.0002),
                "default" : 0
            }
        }
    }
    loss = {
        "type" : "MultiLabelpAUCLoss",
        "space" : {
            "mode" :{
                "val" : "SOPA"
            },
            "margin" : {
                "val" : [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "default" : 1.0
            },
            "beta" :{
                "val" : 0.2
            },
            "eta" : {
                "val" : (0.01, 10),
                "default" : 0.1,
                "log" : True
            }
        }
    }

class pAUC_DRO_LossSpace:
    optimizer = {
        "type" : "SOPAs",
        "space" : {
            "lr" : {
                "val": (0.0001, 0.1),
                "default" : 0.001,
                "log": True
            },
            "momentum" : {
                "val" : (0.8, 0.99),
                "default" : 0.9
            },
            "weight_decay" : {
                "val" : (0.0, 0.0002),
                "default" : 1e-5
            }
        }
    }
    loss = {
        "type" : "pAUCLoss",
        "space" : {
            "mode" :{
                "val" : "SOPAs"
            },
            "gamma": {
                "val" : (0.0, 1.0),
                "default" : 0.9
            },
            "margin" : {
                "val" : [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "default" : 1.0
            },
            "Lambda":{
                "val" : (0.1, 10.0),
                "default" : 1.0,
                "log" : True
            }
        }
    }

class MultiLabelpAUC_DRO_LossSpace:
    optimizer = {
        "type" : "SOPAs",
        "space" : {
            "lr" : {
                "val": (0.0001, 0.1),
                "default" : 0.001,
                "log": True
            },
            "momentum" : {
                "val" : (0.8, 0.99),
                "default" : 0.9
            },
            "weight_decay" : {
                "val" : (0.0, 0.0002),
                "default" : 1e-5
            }
        }
    }
    loss = {
        "type" : "MultiLabelpAUCLoss",
        "space" : {
            "mode" :{
                "val" : "SOPAs"
            },
            "gamma": {
                "val" : (0.0, 1.0),
                "default" : 0.9
            },
            "margin" : {
                "val" : [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "default" : 1.0
            },
            "Lambda":{
                "val" : (0.1, 10.0),
                "default" : 1.0,
                "log" : True
            }
        }
    }

class tpAUC_KL_LossSpace:
    optimizer = {
        "type" : "SOTAs",
        "space" : {
            "lr" : {
                "val": (0.0001, 0.1),
                "default" : 1.0e-3,
                "log": True
            },
            "momentum" : {
                "val" : (0.8, 0.99),
                "default" : 0.9
            },
            "weight_decay" : {
                "val" : (0.0, 0.0002),
                "default" : 0
            }
        }
    }
    loss = {
        "type" : "pAUCLoss",
        "space" : {
            "mode" :{
                "val" : "SOTAs"
            },
            "tau" : {
                "val" : (0.1, 10.0),
                "default" : 1.0,
                "log" : True
            },
            "gammas": {
                "val" : [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)],
                "default" : (0.9, 0.9)
            },
            "margin" : {
                "val" : [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "default" : 1.0
            },
            "Lambda":{
                "val" : (0.1, 10.0),
                "default" : 1.0,
                "log" : True
            }
        }
    }

class tpAUC_CVaR_lossSpace:
    optimizer = {
        "type" : "STACO",
        "space" : {
            "lr" : {
                "val": (0.0001, 0.01),
                "default" : 0.001,
                "log": True
            },
            "momentum" : {
                "val" : (0.8, 0.99),
                "default" : 0.9
            },
            "weight_decay" : {
                "val" : (0.0, 0.0002),
                "default" : 0
            }
        }
    }
    loss = {
        "type" : "tpAUC_CVaR_loss",
        "space" : {
            "threshold" : {
                "val" : [0.3, 0.5, 0.7],
                "default" : 0.5
            },
            "alpha" : {
                "val" : (0.0001, 0.1),
                "default" : 0.1,
                "log" : True
            },
            "beta_0" : {
                "val" : (0.0001, 0.1),
                "default" : 0.1,
                "log" : True
            },
            "beta_1" : {
                "val" : (0.0001, 0.1),
                "default" : 0.1,
                "log" : True
            },
            "theta_0" : {
                "val" : [0.3, 0.5, 0.7],
                "default" : 0.5
            },
            "theta_1" : {
                "val" : [0.3, 0.5, 0.7],
                "default" : 0.5
            }
        }
    }

class MultiLabeltpAUC_KL_LossSpace:
    optimizer = {
        "type" : "SOTAs",
        "space" : {
            "lr" : {
                "val": (0.0001, 0.1),
                "default" : 1.0e-3,
                "log": True
            },
            "momentum" : {
                "val" : (0.8, 0.99),
                "default" : 0.9
            },
            "weight_decay" : {
                "val" : (0.0, 0.0002),
                "default" : 0
            }
        }
    }
    loss = {
        "type" : "MultiLabelpAUCLoss",
        "space" : {
            "mode" :{
                "val" : "SOTAs"
            },
            "tau" : {
                "val" : (0.1, 10.0),
                "default" : 1.0,
                "log" : True
            },
            "gammas": {
                "val" : [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)],
                "default" : (0.9, 0.9)
            },
            "margin" : {
                "val" : [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "default" : 1.0
            },
            "Lambda":{
                "val" : (0.1, 10.0),
                "default" : 1.0,
                "log" : True
            }
        }
    }


class NDCGLossSpace:
    optimizer = {
        "type" : "SONG",
        "space" : {
            "lr" : {
                "val": (0.0001, 0.1),
                "default" : 0.1,
                "log": True
            },
            "momentum" : {
                "val" : (0.8, 0.99),
                "default" : 0.9
            },
            "weight_decay" : {
                "val" : (0.0, 0.0002),
                "default" : 0
            }
        }
    }
    loss = {
        "type" : "NDCGLoss",
        "space" : {
            "gamma0": {
                "val" : (0.0, 1.0),
                "default" : 0.9
            },
            "gamma1": {
                "val" : 0.9
            },
            "eta0" : {
                "val" : (0.001, 0.1),
                "default" : 0.01,
                "log" : True
            },
            "margin" : {
                "val" : [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "default" : 1.0
            },
            "sigmoid_alpha":{
                "val" : (1.0, 2.0),
                "default" : 2.0
            }
        }
    }

class SGDSpace:
    optimizer = {
        "type" : "SGD",
        "space" : {
            "lr" : {
                "val": (0.0001, 0.1),
                "default" : 0.1,
                "log": True
            },
            "momentum" : {
                "val" : [0, 0.9],
                "default" : 0
            },
            "weight_decay" : {
                "val" : (0.0, 0.0002),
                "default" : 0
            }
        }
    }
    loss = {
        "type" : "CrossEntropyLoss",
        "space" : {}
    }

class AdamSpace:
    optimizer = {
        "type" : "Adam",
        "space" : {
            "lr" : {
                "val": (0.0001, 0.1),
                "default" : 0.001,
                "log": True
            },
            "weight_decay" : {
                "val" : (0.0, 0.0002),
                "default" : 0
            }
        }
    }
    loss = {
        "type" : "CrossEntropyLoss",
        "space" : {}
    }

class BCELossSpace:
    loss = {
        "type" : "BCELoss",
        "space" : {}
    }
    optimizer = {}