# coding:utf-8
import numpy as np
import dispy
import time
import threading

__all__ = ['positive_probs_dis', 'design_P_dis', 'design_Q_dis']

master_ip = '10.13.81.186'
node_ip_list = ['10.13.81.184', '10.13.81.185', '10.13.81.186', '10.13.81.187']


def probs_computation(X, W, p_noise):
    import numpy as np
    from scipy import stats
    def positive_probs(X, W, p_noise):
        n_feature, n_hidden = W.shape
        hidden_positive_prob = []
        for i in xrange(n_hidden):
            X_hidden = X * W[:, i]
            mu = np.sum(X_hidden, axis=1) * (1. - p_noise)
            sigma = np.sqrt(np.sum(X_hidden ** 2, axis=1) * (1. - p_noise) * p_noise)
            col_positive_prob = 1. - stats.norm.cdf(-mu / sigma)
            hidden_positive_prob.append(col_positive_prob[:, None])
        hidden_positive_prob = np.concatenate(hidden_positive_prob, axis=1)
        return hidden_positive_prob

    return positive_probs(X, W, p_noise)


def positive_probs_dis(X, W, p_noise):
    result_dis = distribute1(probs_computation, master_ip, node_ip_list, 48, X, W, p_noise)
    result = np.concatenate(result_dis, axis=1)
    return result


def Q_computation(X, W, P_positive, p_noise):
    import numpy as np
    def add_Q_noise(S_X, p_noise):
        n_feature = S_X.shape[0]
        S_X *= (1. - p_noise) ** 2
        diag_idx = np.diag_indices(n_feature - 1)
        S_X[diag_idx] /= 1. - p_noise
        S_X[-1, :-1] /= 1. - p_noise
        S_X[:-1, -1] /= 1. - p_noise
        S_X[-1, -1] /= (1. - p_noise) ** 2
        return S_X

    def design_Q(X, W, P_positive, p_noise):
        n_batch, n_feature = X.shape
        n_feature, n_hidden = W.shape
        Q = np.zeros((n_hidden, n_hidden), dtype=float)
        for i in xrange(n_batch):
            X_row = X[[i], :]
            S_X = np.dot(X_row.T, X_row)
            S_X = add_Q_noise(S_X, p_noise)
            W_p = W * P_positive[i, :]
            half = np.dot(W_p.T, S_X)
            Q_i = []
            for j in xrange(n_hidden):
                Q_i.append(np.dot(half, W_p[:, [j]]))
            Q_i = np.concatenate(Q_i, axis=1)
            Q += Q_i
        return Q

    return design_Q(X, W, P_positive, p_noise)


def design_Q_dis(X, W, P_positive, p_noise):
    result_dis = distribute_efficient(Q_computation, master_ip, node_ip_list, 100, X, W, P_positive, p_noise)
    result = sum(result_dis.values())
    return result


def P_computation(X, W, P_positive, p_noise):
    import numpy as np
    def add_P_noise(S_X, p_noise):
        S_X *= 1. - p_noise
        S_X[-1, :] /= 1. - p_noise
        return S_X

    def design_P(X, W, P_positive, p_noise):
        n_batch, n_feature = X.shape
        n_feature, n_hidden = W.shape
        P = np.zeros((n_hidden, n_feature), dtype=float)
        for i in xrange(n_batch):
            X_row = X[[i], :]
            S_X = X_row.T.dot(X_row)
            S_X = add_P_noise(S_X, p_noise)
            P_row = P_positive[i, :]
            P += np.dot((W * P_row).T, S_X)
        return P

    return design_P(X, W, P_positive, p_noise)


def design_P_dis(X, W, P_positive, p_noise):
    result_dis = distribute_efficient(P_computation, master_ip, node_ip_list, 100, X, W, P_positive, p_noise)
    result = sum(result_dis.values())
    return result


########################################################################################################################


def deploy(batch, n_jobs):
    dis_batch = (batch // n_jobs) * np.ones(n_jobs, dtype=np.int)
    dis_batch[:batch % n_jobs] += 1
    starts = np.cumsum(dis_batch)
    starts = [0] + starts.tolist()
    return starts


def distribute1(function, master_ip, node_ip_list, n_jobs, *dis_args):
    cluster = dispy.JobCluster(function, ip_addr=master_ip, nodes=node_ip_list)
    X, W, p_noise = dis_args
    starts = deploy(W.shape[1], n_jobs)
    jobs = []
    for i in range(n_jobs):
        job = cluster.submit(X, W[:, starts[i]:starts[i + 1]], p_noise)  # X很大网络带宽不能承受
        job.id = i
        jobs.append(job)
        time.sleep(20)  # 延时发送数据,网络带宽不足会分配失败
    cluster.wait()
    results = []
    for job in jobs:
        job_result = job()
        results.append(job_result)
        print 'finish train_lin_job %d' % job.id
    cluster.close()
    return results


def distribute2(function, master_ip, node_ip_list, n_jobs, *dis_args):
    cluster = dispy.JobCluster(function, ip_addr=master_ip, nodes=node_ip_list)
    X, W, P_positive, p_noise = dis_args
    starts = deploy(X.shape[0], n_jobs)
    jobs = []
    for i in range(n_jobs):
        job = cluster.submit(X[starts[i]:starts[i + 1]], W, P_positive[starts[i]:starts[i + 1]], p_noise)
        job.id = i
        jobs.append(job)
        time.sleep(20)  # 延时发送数据,网络带宽不足会分配失败
    cluster.wait()
    results = []
    for job in jobs:
        job_result = job()
        results.append(job_result)
        print 'finish train_lin_job %d' % job.id
    cluster.close()
    return results


def job_callback(job):  # executed at the client
    global pending_jobs, jobs_cond, lower_bound, results
    if (job.status == dispy.DispyJob.Finished  # most usual case
        or job.status in (dispy.DispyJob.Terminated, dispy.DispyJob.Cancelled,
                          dispy.DispyJob.Abandoned)):
        # 'pending_jobs' is shared between two threads, so access it with
        # 'jobs_cond' (see below)
        jobs_cond.acquire()
        if job.id:  # train_lin_job may have finished before 'main' assigned id
            pending_jobs.pop(job.id)
            results[job.id] = job.result
            print 'finish train_lin_job', job.id
            # dispy.logger.info('train_lin_job "%s" done with %s: %s', train_lin_job.id, train_lin_job.result, len(pending_jobs))
            if len(pending_jobs) <= lower_bound:
                jobs_cond.notify()
        jobs_cond.release()


def distribute_efficient(function, master_ip, node_ip_list, n_jobs, *dis_args):
    X, W, P_positive, p_noise = dis_args
    starts = deploy(X.shape[0], n_jobs)
    global pending_jobs, jobs_cond, lower_bound, results
    # set lower and upper bounds as appropriate; assuming there are 30
    # processors in a cluster, bounds are set to 50 to 100
    lower_bound, upper_bound = 10, 20
    # use Condition variable to protect access to pending_jobs, as
    # 'job_callback' is executed in another thread
    jobs_cond = threading.Condition()
    cluster = dispy.JobCluster(function, ip_addr=master_ip, nodes=node_ip_list, callback=job_callback)
    pending_jobs = {}
    results = {}
    for i in range(n_jobs):
        job = cluster.submit(X[starts[i]:starts[i + 1]], W, P_positive[starts[i]:starts[i + 1]], p_noise)
        jobs_cond.acquire()
        job.id = i
        # there is a chance the train_lin_job may have finished and job_callback called by
        # this time, so put it in 'pending_jobs' only if train_lin_job is pending
        if job.status == dispy.DispyJob.Created or job.status == dispy.DispyJob.Running:
            pending_jobs[i] = job
            # dispy.logger.info('train_lin_job "%s" submitted: %s', i, len(pending_jobs))
            if len(pending_jobs) >= upper_bound:
                while len(pending_jobs) > lower_bound:
                    jobs_cond.wait()
        jobs_cond.release()
    cluster.wait()
    cluster.close()
    return results
