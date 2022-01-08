from multiprocessing import freeze_support
from dask.distributed import Client, LocalCluster
import dask.array as da



if __name__ == '__main__':
    freeze_support()
    cluster = LocalCluster(n_workers=4)
    client = Client(cluster)
    print(client.dashboard_link)

    array = da.ones((10000, 1000, 1000), chunks=100)
    print(array.mean().compute())  # Should print 1.0

    print('all done')