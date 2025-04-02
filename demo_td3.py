from algorithms.td3 import TD3
from demo import TorchDemo

td3 = TD3()
td3.load_policy("./out/td3.pth")

demo = TorchDemo(td3.policy)
demo.run()