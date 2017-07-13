"""Script that demonstrates creating a class instance across a cluster using a manager.

    The instance can be accessed and modified concurrently across the cluster using its lock attribute.

"""
import time
import Queue
from multiprocessing import Process, Lock
from multiprocessing.managers import BaseManager


class WorkManager(BaseManager):
    pass


class SomeClass(object):
    def __init__(self, val):
        self.val = val
        self.lock = Lock()

    def inc_val(self, number):
        with self.lock:
            self.val += number
            print self.val
            time.sleep(1)

    def get_val(self):
        return self.val


def continuous_update(var):
    for i in range(10):
        var.inc_val(1)


def server_manager():

    # Start manager on port 55555
    manager = WorkManager(address=('', 55555), authkey='password')
    manager.start()
    print "Server started on port 55555"

    # Access manager's objects
    shared_in_q = manager.in_q()
    shared_out_q = manager.out_q()
    shared_class_var = manager.class_var()

    # Start process to update shared class variable
    p = Process(target=continuous_update, args=(shared_class_var,))
    p.start()

    # Create work in shared input queue
    for i in range(10):
        shared_in_q.put(i)

    # Wait for results in shared output queue
    total_results = 10
    result_list = []
    while len(result_list) < total_results:
        result = shared_out_q.get()
        # print result, shared_class_var.get_val()
        result_list.append(result)

    # Join continuous process
    p.join()

    # Sleep before shutting down server to give clients time to exit
    time.sleep(2)
    manager.shutdown()


if __name__ == '__main__':

    # Create shared instances
    in_q = Queue.Queue()
    out_q = Queue.Queue()
    class_var = SomeClass(0)

    # Register shared instances with manager class
    WorkManager.register('class_var', callable=lambda: class_var)
    WorkManager.register('in_q', callable=lambda: in_q)
    WorkManager.register('out_q', callable=lambda: out_q)

    # Run main script
    server_manager()
