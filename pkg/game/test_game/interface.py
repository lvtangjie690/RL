# coding=utf-8
from multiprocessing.managers import BaseManager
import threading

from . import config


class Interface(object):
    def __init__(self, id):
        self.id = id

        self.action = None

        BaseManager.register('Game2Agent')
        BaseManager.register('Agent2Game')
        manager = BaseManager(address=(config.server, config.server_port + self.id), authkey='rl',serializer='xmlrpclib')
        manager.connect()
        print("port : {}".format(config.server_port + self.id))

        self.g2a = manager.Game2Agent()
        self.a2g = manager.Agent2Game()

    # thread = threading.Thread(target=self.on_receive)
    # thread.setDaemon(True)
    # thread.start()

    def ReceiveAction(self):
        self.action = self.a2g.get()

    # Lua Interface
    def SendSample(self, state, action, reward, next, done):
        sample = [LuaTable2List(state), action, reward, LuaTable2List(next), done]
        self.g2a.put(sample)


def LuaTable2List(lua_table):
    if not lua_table or type(lua_table) == list:
        return lua_table
    else:
        return [float(lua_table[x + 1]) for x in range(len(lua_table))]
