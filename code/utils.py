import time

def timer(f):
    def wrapper_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print('Timer: {} took {} seconds'.format(f.__name__, end - start))
        return result
    return wrapper_timer