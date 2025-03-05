import array
import psutil
import subprocess
import sys
import numpy as np
import os
from py4j.java_gateway import JavaGateway, java_import


def init_gateway():
    gateway = JavaGateway()

    java_import(gateway.jvm, 'java.lang.int')
    java_import(gateway.jvm, 'java.util.Arrays')

    greeting = gateway.entry_point.getGreeting()

    if greeting != "Hello, World!":
        breakpoint()

    return gateway

# https://stackoverflow.com/questions/39095994/fast-conversion-of-java-array-to-numpy-array-py4j # noqa
# NOTE: ints only
def create_2d_java_array(numpy_matrix, gateway):
   header = array.array('i', list(numpy_matrix.shape))
   body = array.array('i', numpy_matrix.flatten().tolist());
   if sys.byteorder != 'big':
      header.byteswap()
      body.byteswap()
   buf = bytearray(header.tobytes() + body.tobytes())
   return gateway.entry_point.createFromPy4j(buf)

# NOTE: doesn't use bytearray for fast interop
def create_1d_java_array(numpy_array, gateway):
    jarray = gateway.new_array(gateway.jvm.int, numpy_array.shape[0])
    for i in range(numpy_array.shape[0]):
        jarray[i] = int(numpy_array[i])
    return jarray

def fit_hdp(data, gateway):
    # NOTE: java HDP considers first column as class
    data2 = np.c_[data[:, -1], data[:, :-1]]
    jdata = create_2d_java_array(data2, gateway)
    # swap class to first column
    tree = gateway.entry_point.train_hdp(jdata)
    return tree

def query_hdp(tree, query, gateway):
    jquery = create_1d_java_array(query, gateway)
    return np.array(tree.query(jquery))

def probs_hdp(tree, strings, gateway):
    first = query_hdp(tree, strings[0], gateway)
    card = first.shape[0]
    probs = np.zeros((strings.shape[0], card))
    probs[0] = first
    for i in range(1, strings.shape[0]):
        probs[i] = query_hdp(tree, strings[i], gateway)
    return probs

def launch_java_process(command):
    process = subprocess.Popen(command, cwd=os.path.dirname(os.path.realpath(__file__)) )
    return process


def check_process_status(process):
    if process is None:
        return "Process not found"

    if isinstance(process, subprocess.Popen):
        if process.poll() is None:
            return "Running"
        else:
            return f"Terminated with return code {process.returncode}"
    elif isinstance(process, int):
        try:
            p = psutil.Process(process)
            return p.status()
        except psutil.NoSuchProcess:
            return "Process not found"
    else:
        return "Invalid process object"

def stop_process(process):
    if process is None:
        print("No process to stop")
        return

    if isinstance(process, subprocess.Popen):
        pid = process.pid
    elif isinstance(process, int):
        pid = process
    else:
        print("Invalid process object")
        return

    try:
        p = psutil.Process(pid)
        p.terminate()
        print(f"Process with PID {pid} terminated")
    except psutil.NoSuchProcess:
        print(f"Process with PID {pid} not found")
    except Exception as e:
        print(f"Error stopping process: {e}")

