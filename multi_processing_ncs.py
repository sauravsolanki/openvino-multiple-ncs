from openvino.inference_engine import IENetwork, IEPlugin
import cv2
import multiprocessing as mp
import os
from sys import argv
import datetime

#global variables 
root="./models/frozen_inference_graph.%s"
run_async = False
num_devices = 1


def handle_args():
    """Reads the commandline args and adjusts initial values of globals values to match

    :return: False if there was an error with the args, or True if args processed ok.
    """
    global num_devices, run_async
    
    for an_arg in argv:
        lower_arg = str(an_arg).lower()
        if (an_arg == argv[0]):
            continue

        elif (lower_arg == 'help'):
            return False

        elif (lower_arg.startswith('num_devices=') or lower_arg.startswith("nd=")):
            try:
                arg, val = str(an_arg).split('=', 1)
                num_devices = int(val)
                if (num_devices < 0):
                    print('Error - num_devices argument invalid.  It must be > 0')
                    return False
                print('setting num_devices: ' + str(num_devices))
            except:
                print('Error - num_devices argument invalid.  It must be between 1 and number of devices in system')
                return False

        elif (lower_arg.startswith('run_async=') or lower_arg.startswith("ra=")):
            try:
                arg, val = str(an_arg).split('=', 1)
                run_async = True if val == "true" else False
            except:
                print('Run Aysnc Error!')
                return False

def print_arg_vals():
    global num_devices, run_async
    print("")
    print("--------------------------------------------------------")
    print("Current date and time: " + str(datetime.datetime.now()))
    print("")
    print("program arguments:")
    print("------------------")
    print('device: MYRIAD' )
    print('num_devices: ' + str(num_devices))
    print("Running Mode: ",run_async)
    print("--------------------------------------------------------")   

def start():
    global num_devices, run_async,root

    plugin = IEPlugin(device="MYRIAD")
    net = IENetwork(root%'xml',root%'bin')
        
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    exec_net = plugin.load(network=net)
    
    frame = cv2.imread("./traffic.jpg")
    n, c, h, w = net.inputs[input_blob].shape
    image = cv2.resize(frame, (w, h))
    image = image.transpose((2, 0, 1))
    image = image.reshape((n, c, h, w))

    if not run_async:
        print("running in sync mode")
    else:
        print("running in async mode")

    while(not run_async):
        res = exec_net.infer({input_blob: image})
        output_blob = res
        print("",os.getpid(),output_blob[out_blob].shape)

    while run_async:
        #async operation
        exec_net.start_async(0,{input_blob: image})
        while(True):
            status = exec_net.requests[0].wait(-1)
            if status == 0:
                break
            else:
                time.sleep(1)
        res = exec_net.requests[0].outputs
        print("",os.getpid(),res[out_blob].shape)


#####################################################

def run():
    global num_devices   
    processes = []
    handle_args()
    print_arg_vals()
    nets = [None]*num_devices    
    for net in nets:
        processes.append(mp.Process(target=start, args=()))
    
    [setattr(process, "daemon", True) for process in processes]  # process.daemon = True
    [process.start() for process in processes]
    [process.join() for process in processes]
    del nets

try:
    run()
except KeyboardInterrupt as e:
    print("Closed By user")
except Exception as e:
    print("Unknown Error")
    print(e)
finally:
    handle_args()
