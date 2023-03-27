# import snap7
from threading import *
import time

from snap7.exceptions import Snap7Exception

import src.pythonsnap7.snap7.client as c
from src.pythonsnap7.snap7.util import *
from src.pythonsnap7.snap7.types import *


class PLCSiemens:
    def __init__(self, ip, rack, slot):
        self.name = "PLC dummy"
        self.l = Lock()
        self.run = False
        self.IP = ip
        self.RACK = rack
        self.SLOT = slot
        self.plc = None
        self.plc_2 = None
        self.auto_mode = False
        self.artificial_vision = False
        self.trigger = False
        self.full_mode = False
        self.camara1x = 0
        self.camara1y = 0
        self.camara2x = 0
        self.camara2y = 0

    def start(self):
        print("PLC Connecting...")
        self.plc = c.Client()
        self.plc.connect(self.IP, self.RACK, self.SLOT)

        self.plc_2 = c.Client()
        self.plc_2.connect(self.IP, self.RACK, self.SLOT)
        self.run = True

    def connect(self):
        while True:
            # check connection
            if self.plc.get_connected():
                break
            try:
                # attempt connection
                self.plc.connect(self.IP, self.RACK, self.SLOT)
            except:
                pass
            time.sleep(2.5)

    def StartGrabbing(self):
        while self.run:
            self.l.acquire()
            # Reading Mem from PLC
            try:
                self.auto_mode = self.ReadMemory(self.plc, 720, 1, S7WLBit)
                self.trigger = self.ReadMemory(self.plc, 720, 2, S7WLBit)
                self.artificial_vision = self.ReadMemory(self.plc, 720, 3, S7WLBit)
                self.full_mode = self.ReadMemory(self.plc, 790, 1, S7WLBit)
                self.camara1x = self.ReadMemory(self.plc, 810, 0, S7WLWord)
                self.camara1y = self.ReadMemory(self.plc, 812, 0, S7WLWord)
                self.camara2x = self.ReadMemory(self.plc, 814, 0, S7WLWord)
                self.camara2y = self.ReadMemory(self.plc, 816, 0, S7WLWord)
            except Snap7Exception as e:
                print("PLC connecting...")
                time.sleep(2.5)
                #self.connect()
            self.l.release()
            time.sleep(0.001)
        self.plc.disconnect()

    def ReadMemory(self, plc, byte, bit, datatype):
        result = plc.read_area(areas['MK'], 0, byte, datatype)
        if datatype == S7WLBit:
            return get_bool(result, 0, bit)
        elif datatype == S7WLByte or datatype == S7WLWord:
            return get_int(result, 0)
        elif datatype == S7WLReal:
            return get_real(result, 0)
        elif datatype == S7WLDWord:
            return get_dword(result, 0)
        else:
            return None

    def WriteMemory(self, plc, byte, bit, datatype, value):
        result = plc.read_area(areas['MK'], 0, byte, datatype)
        if datatype == S7WLBit:
            set_bool(result, 0, bit, value)
        elif datatype == S7WLByte or datatype == S7WLWord:
            set_int(result, 0, value)
        elif datatype == S7WLReal:
            set_real(result, 0, value)
        elif datatype == S7WLDWord:
            set_dword(result, 0, value)
        plc.write_area(areas["MK"], 0, byte, result)

    def write_defects(self, output):
        if self.run:
            # Figure 1
            self.WriteMemory(self.plc_2, 703, 0, S7WLBit, output[0])
            # Figure 2
            self.WriteMemory(self.plc_2, 703, 1, S7WLBit, output[1])
            # Figure 3
            self.WriteMemory(self.plc_2, 703, 2, S7WLBit, output[2])
            # Figure 4
            self.WriteMemory(self.plc_2, 703, 3, S7WLBit, output[3])
            # Figure 5
            self.WriteMemory(self.plc_2, 703, 4, S7WLBit, output[4])
            # Figure 6
            self.WriteMemory(self.plc_2, 703, 5, S7WLBit, output[5])
            # Figure 7
            self.WriteMemory(self.plc_2, 703, 6, S7WLBit, output[6])
            # Figure 8
            self.WriteMemory(self.plc_2, 703, 7, S7WLBit, output[7])
            # Figure 9
            self.WriteMemory(self.plc_2, 702, 0, S7WLBit, output[8])
            # Figure 10
            self.WriteMemory(self.plc_2, 702, 1, S7WLBit, output[9])
            # Figure 11
            self.WriteMemory(self.plc_2, 702, 2, S7WLBit, output[10])
        else:
            print("Unable to Write output to the PLC.")

    def write_flag(self, flag):
        if self.run:
            self.WriteMemory(self.plc_handler.plc_2, 720, 5, S7WLBit, flag)
        else:
            print("Unable to Write Flag to the PLC.")

    def stop(self):
        print("PLC DissConnecting...")
        if self.run:
            self.plc_2.disconnect()
        self.run = False
