# coding:utf-8
import numpy as np
# 'compute' is distributed to each node running 'dispynode'
def compute(n):
    import time, socket
    time.sleep(n)
    host = socket.gethostname()
    return (host, n)

def distribute(function, master_ip, node_ip_list, n_jobs, dis_arg, dis_dim, *rest_args):
    cluster = dispy.JobCluster(function, ip_addr=master_ip, nodes=node_ip_list)
    batch = dis_arg.shape[dis_dim]
    dis_batch = (batch // n_jobs) * np.ones(n_jobs, dtype=np.int)
    dis_batch[:batch % n_jobs] += 1
    starts = np.cumsum(dis_batch)
    starts = [0] + starts.tolist()
    jobs = []
    for i in range(n_jobs):
        job = cluster.submit(np.take(dis_arg, range(starts[i], starts[i + 1]), axis=dis_dim), *rest_args)
        job.id = i
        jobs.append(job)
    results = []
    for job in jobs:
        results.append(job())
        print 'finish train_lin_job %d' % job.id
    return results

if __name__ == '__main__':
    import dispy, random
    cluster = dispy.JobCluster(compute)
    jobs = []
    for i in range(10):
        # schedule execution of 'compute' on a node (running 'dispynode')
        # with a parameter (random number in this case)
        job = cluster.submit(random.randint(5,20))
        job.id = i # optionally associate an ID to train_lin_job (if needed later)
        jobs.append(job)
    cluster.wait() # waits for all scheduled jobs to finish
    for job in jobs:
        host, n = job() # waits for train_lin_job to finish and returns results
        print('%s executed train_lin_job %s at %s with %s' % (host, job.id, job.start_time, n))
        # other fields of 'train_lin_job' that may be useful:
        # print(train_lin_job.stdout, train_lin_job.stderr, train_lin_job.exception, train_lin_job.ip_addr, train_lin_job.start_time, train_lin_job.end_time)
    cluster.print_status()