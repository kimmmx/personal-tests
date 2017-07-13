"""Script that demonstrates accessing a server class instance across a cluster using a manager.

    The instance can be accessed and modified concurrently across the cluster using its lock attribute.

"""
import time
import Queue
from multiprocessing.managers import BaseManager


class WorkManager(BaseManager):
    pass


def client_manager():

    # Connect to manager
    manager = WorkManager(address=('192.168.0.41', 55555), authkey='password')
    manager.connect()
    print "Client connected to 192.168.0.41:55555"

    # Access manager's objects
    shared_in_q = manager.in_q()
    shared_out_q = manager.out_q()
    shared_class_var = manager.class_var()

    # Get work from shared input queue
    while True:
        try:
            in_var = shared_in_q.get(timeout=0.1)
            # shared_class_var.inc_val(1)
            a = shared_class_var.get_val()
            print a
            time.sleep(2)
            shared_out_q.put(in_var)
        except Queue.Empty:
            break


if __name__ == '__main__':

    # Register shared instances with manager class
    WorkManager.register('class_var')
    WorkManager.register('in_q')
    WorkManager.register('out_q')

    # Run main script
    client_manager()
