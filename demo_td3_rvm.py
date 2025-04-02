from algorithms.td3_rvm import TD3RVM
from demo import TorchDemo

td3 = TD3RVM()
td3.load_policy("./out/td3_rvm.pth")

demo = TorchDemo(td3.policy)
demo.run()