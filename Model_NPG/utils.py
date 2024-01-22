import torch
import torch.nn.functional as F

import numpy as np

def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.contiguous().view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten)
    return hessians_flatten


# def flat_params(model):
#     params = []
#     for param in model.parameters():
#         params.append(param.data.view(-1))
#     params_flatten = torch.cat(params)
#     return params_flatten

def flat_params(model_params):
    params = []
    for param in model_params:
        params.append(param.contiguous().view(-1).data)
    params_flatten = torch.cat(params)
    return params_flatten.clone()

# def update_model(model, new_params):
#     index = 0
#     for params in model.parameters():
#         params_length = len(params.view(-1))
#         new_param = new_params[index: index + params_length]
#         new_param = new_param.view(params.size())
#         params.data.copy_(new_param)
#         index += params_length
        
def update_model(model_params, new_params):
    index = 0
    for params in model_params:
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length


# def set_param_values(model_params, new_params, param_sizes, param_shapes, min_log_std, device, *args, **kwargs):
#     current_idx = 0
#     for idx, param in enumerate(model_params):
#         vals = new_params[current_idx:current_idx + param_sizes[idx]]
#         vals = vals.reshape(param_shapes[idx])
#         # clip std at minimum value
#         vals = torch.max(vals, min_log_std) if idx == 0 else vals
#         param.data = vals.to(device).clone()
#         current_idx += param_sizes[idx]

def conjugate_gradient(policy, states, b, x0, action_size, device, iters = 25, residual_tol = 1e-10):
    x = x0 #torch.zeros(b.size(), device = device)
    r = b - fisher_vector_product(policy, states, x, action_size)
    p = r.clone() #b.clone()
    
    for i in range(iters):
        rdotr = torch.dot(r, r)
        _Avp = fisher_vector_product(policy, states, p, action_size)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        
        if new_rdotr < residual_tol:
            break
    return x

# def fisher_vector_product(policy, states, p, action_size):
#     p.detach()
#     kl = kl_div(new_policy = policy, old_policy = policy, states = states, action_size = action_size)
#     kl = kl.mean()
#     kl_grad = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
#     kl_grad = flat_grad(kl_grad)  # check kl_grad == 0

#     kl_grad_p = (kl_grad * p).sum()
#     kl_hessian_p = torch.autograd.grad(kl_grad_p, policy.parameters())
#     kl_hessian_p = flat_hessian(kl_hessian_p)

#     return kl_hessian_p + 1e-4 * p

def fisher_vector_product(policy, states, p, action_size):
    kl = kl_div(new_policy = policy[0], old_policy = policy[0], log_std = policy[2],
                states = states, action_size = action_size)
    kl = kl.mean()
    kl_grad = torch.autograd.grad(kl, policy[1], create_graph=True)
    kl_grad = flat_grad(kl_grad)  # check kl_grad == 0

    kl_grad_p = (kl_grad * p).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, policy[1])
    kl_hessian_p = flat_hessian(kl_hessian_p)

    return kl_hessian_p + 1e-4 * p

# def kl_div(new_policy, old_policy, states, action_size):
#     output_new = new_policy(states)
#     mean, log_var = output_new[:, :, :action_size], output_new[:, :, action_size:]
        
#     output_old = old_policy(states)
#     mean_old, log_var_old = output_old[:, :, :action_size], output_old[:, :, action_size:]
         
#     std, std_old = log_var.exp().sqrt(), log_var_old.exp().sqrt()
#     log_std, log_std_old = log_var.exp().sqrt().log(), log_var_old.exp().sqrt().log()
     
#     mean_old = mean_old.detach()
#     std_old = std_old.detach()
#     log_std_old = log_std_old.detach()

#     kl = -log_std_old + log_std + (std_old.pow(2) + (mean_old - mean).pow(2)) / \
#          (2.0 * std.pow(2)) - 0.5
#     return kl.sum(-1)

def kl_div(new_policy, old_policy, log_std,
           states, action_size):
    
    mean = new_policy(states)
        
    with torch.no_grad():
        mean_old = old_policy(states)
        
    log_std_old = log_std.detach().clone()
         
    std, std_old = log_std.exp(), log_std_old.exp()
     
    # mean_old = mean_old.detach()
    # log_std_old = log_std.detach().clone()

    # kl = log_std_old - log_std + (std_old.pow(2) + (mean_old - mean).pow(2)) / \
    #       (2.0 * std.pow(2)) - 0.5
    
    # This seemed to help with the numerical issues along with input stand. in policy and value
    Nr = (mean_old - mean) ** 2 + std_old ** 2 - std ** 2
    Dr = 2 * std ** 2 + 1e-8
    sample_kl = torch.sum(Nr / Dr + log_std - log_std_old, dim=-1)
    return sample_kl # kl.sum(-1)
