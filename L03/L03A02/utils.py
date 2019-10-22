#!/usr/bin/nv python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:26:06 2019

@author: root
"""

def metric(y_true,y_pred):
    return (abs(y_true-y_pred)).mean()
