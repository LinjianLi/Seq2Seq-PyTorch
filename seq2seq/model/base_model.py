#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

# Modified by Linjian Li

import os
import torch
import torch.nn as nn
# from prettytable import PrettyTable


class BaseModel(nn.Module):
    """
    BaseModel
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *input):
        """
        forward
        """
        raise NotImplementedError

    def __repr__(self):
        # table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            num_param = parameter.numel()
            # table.add_row([name, num_param])
            total_params += num_param
        main_string = '\n' + super(BaseModel, self).__repr__() + '\n'
        # main_string += table.get_string() + '\n'
        main_string += "Total Trainable Params: {}".format(total_params)
        return main_string

    def save(self, filename):
        """
        save
        """
        torch.save(self.state_dict(), filename)
        print("Saved model state to '{}'!".format(filename))

    def load(self, filename):
        """
        load
        """
        if os.path.isfile(filename):
            state_dict = torch.load(
                filename, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict, strict=False)
            print("Loaded model state from '{}'".format(filename))
        else:
            print("Invalid model state file: '{}'".format(filename))
            raise FileNotFoundError
