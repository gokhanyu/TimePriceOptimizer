import os
import imp
import ctypes
import _thread
import win32api

#FIX FOR SCIPY CTRL+C INTERRUPT HANDLING WHICH PREVENTS PYTHON KERNEL CRASH IN WINDOWS
#SET BELOW DLL FOLDER
class FixInterruptHandling():

	def fix(self) : 

		# Load the DLL manually to ensure its handler gets
		# set before our handler.
		basepath = imp.find_module('numpy')[1]
		basepath = 'd:\\ProgramData\\Anaconda3\\Library\\bin\\' ##SET libmmd.dll PATH HERE!

		ctypes.CDLL(os.path.join(basepath, 'libmmd.dll'))
		ctypes.CDLL(os.path.join(basepath, 'libifcoremd.dll'))

		# Now set our handler for CTRL_C_EVENT. Other control event 
		# types will chain to the next handler.
		def handler(dwCtrlType, hook_sigint=_thread.interrupt_main):
				if dwCtrlType == 0: # CTRL_C_EVENT
						hook_sigint()
						return 1 # don't chain to the next handler
				return 0 # chain to the next handler

		win32api.SetConsoleCtrlHandler(handler, 1)
