import argparse
from config.config_uof import Params
from agent.universal_option_framework import UniversalOptionFramework as UOF

parser = argparse.ArgumentParser()

agent = UOF(params=Params)
agent.run()