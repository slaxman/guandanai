import copy
import time

import json
from .state import State
from .action import Action
import random

class ruleAgent():
    def __init__(self, index) -> None:
        self.state = State("client"+str(index))
        self.action = Action()
        self.index = index

    def preprocess_noisy_message(self, message_list):
        received_message = []
        for message in message_list:
            if message['pos'] == self.index:
                received_message.append(copy.deepcopy(message['msg']))
        return received_message


    def reset(self):
        self.state = State("client"+str(self.index))
        self.action = Action()
        #print('reset, ', self.index)

    def decode(self, message_list, tri_back_mode=False):
        #print(message_list)
        message_list = self.preprocess_noisy_message(message_list[:-1])
        #print(self.index, message_list)
        start_pos = 0
        if tri_back_mode:
            for index, message in enumerate(message_list):
                if message['stage'] == 'beginning':
                    start_pos = index
                    break
        #assert len(message_list) > 0
        #print(start_pos, message_list)
        for message in message_list[start_pos:]:
            if message['stage'] == 'afterTri':
                continue
            if message['stage'] == 'tribute':
                pass
            self.state.parse(message, self.index)

            #print('pos %d decode message' % self.index, message )
            #print('res', self.index, self.state.tribute_result)

    def step(self, message):
        message = message[-1]
        # print(message)
        # print(message['pos'],self.index)
        assert message['pos'] == self.index
        self.state.parse(message['msg'])
        assert message['msg']['type'] == 'act'
        act_index = self.action.parse_AI(message['msg'], self.state._myPos)
        return act_index


# class EvalRuleAgent(ruleAgent):
#
#     def decode(self, message_list, tri_back_mode=False):
#         return
#
#     def observe(self, message_list):
#         for message in message_list:
#             if message['stage'] in ['afterTri', 'player_card', 'update_hand']:
#                 assert message['type'] == 'notify'
#                 continue
#             self.state.parse(message)
#
#     def step(self, state):
#         message_list = state.message_list
#         for message in message_list:
#
#             if message['stage'] in ['afterTri', 'player_card', 'update_hand']:
#                 assert message['type'] == 'notify'
#                 continue
#             # print('!!!!!', message)
#             self.state.parse(message)
#             if message['type'] == 'act':
#                 action_index = self.action.rule_parse(message,
#                                                       copy.deepcopy(self.state.myPos), copy.deepcopy(self.state.remain_cards), copy.deepcopy(self.state.history),
#                                                       copy.deepcopy(self.state.remain_cards_classbynum), copy.deepcopy(self.state.pass_num),
#                                                       copy.deepcopy(self.state.my_pass_num), copy.deepcopy(self.state.tribute_result))
#                 return action_index
#         return
