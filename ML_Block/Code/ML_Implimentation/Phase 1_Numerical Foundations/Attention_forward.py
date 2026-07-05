import numpy as np



def softmax(S):
    max_ = np.max(S, axis= -1, keepdims= True)
    S_ = S-max_
    S_SoftMax = np.exp(S_)/np.sum(np.exp(S_), axis= -1, keepdims=True)

    return S_SoftMax

def mask_attention(Q, K, V):
    S =  Q @ np.transpose(K, (0, 2, 1))
    b, r, c = S.shape
    maks_ = np.tril(np.ones((r, c), dtype=bool))
    S_mask = np.where(maks_ == 1, S, -1e10)
    S_SoftMax = softmax(S_mask)
    A = S_SoftMax @ V
    return  A


def attention (Q, K, V):
    # b, t, d
    d = V.shape[-1]
    S = Q @ np.transpose(K,(0, 2, 1))*(1./np.sqrt(d)) # (n, t, t)
    SoftMax = softmax(S)
    Out = SoftMax @ V
    cash = (SoftMax, Out)
    return   cash
def backprob_attention(dOut, Q, K, V):
     d = K.shape [-1]
     S = (Q @ np.transpose(K, (0, 2, 1)))*(1./np.sqrt(d))
     cash = attention(Q, K, V)
     P, Out = cash
     # P = softmax(q,k), out = P.V
     dP = dOut @ np.transpose(V, (0,2,1))
     dV = np.transpose(P, (0, 2, 1)) @ dOut
     dS = P * (dP-np.sum(dP*P, axis= -1, keepdims= True))
     dQ = (dS @ K) *(1./np.sqrt(d))
     dK = (np.transpose(dS, (0, 2, 1)) @ Q) *(1./np.sqrt(d))
     return dV, dK, dQ

def sanity_check(R, Q, K, V, Eps):
    parameters_list = [Q, K, V]
    ans = []
    for parameters in parameters_list:
        b, r, c = parameters.shape
        dr_parameters = parameters.copy()
        for i in range (b):
          for j in range(r):
             for k in range(c):
                origin  = parameters[i][j][k]
                parameters[i][j][k] =  parameters[i][j][k] + Eps
                SoftMax, Out = attention(*parameters_list)
                L2 = np.sum(Out * R)
                parameters[i][j][k] = origin
                parameters[i][j][k] = parameters[i][j][k] - Eps
                SoftMax, Out = attention(*parameters_list)
                L1 = np.sum(Out * R)
                dr_parameters [i][j][k] = (L2 - L1) / (2*Eps)
                parameters[i][j][k] = origin
        ans.append(dr_parameters)
    return ans

def multi_head_attention_forward(Q, K, V, num_heads):
    #Q, k, v # b, h, t, head_d
    d = Q.shape[-1]
    head_d = d// num_heads
    b = Q.shape[0]; t = Q.shape[1]
    #b =0, t =1, num_heads =2 , head_d= 3
    Q = Q.reshape((b, t, num_heads, head_d)); Q = np.transpose(Q, (0, 2, 1,  3))
    K = K.reshape((b, t, num_heads, head_d)); K = np.transpose(K, (0, 2, 1,  3))
    V = V.reshape((b, t, num_heads, head_d)); V = np.transpose(V, (0, 2, 1,  3))
    KT = np.transpose(K, (0, 1, 3, 2))
    S = (Q @ KT)*1./np.sqrt(head_d)
    X_Max = np.max(S, axis= -1, keepdims= True)
    P = np.exp(S-X_Max)/np.sum(np.exp(S-X_Max), axis= -1 , keepdims= True) # b, h, t, t
    Out = P @ V  # b, h, t, head_d
    Out = np.transpose(Out, (0, 2, 1,  3))
    Out = Out.reshape((b, t, d))
    cash = (P, Q, K, V, num_heads, head_d)
    return Out, cash
def backward_multihead_attention(dOut, cash ):
    P, Q, K, V, num_heads, head_d = cash

    # P = b, h, t, t
    # Q, K, V = b, h, t, d_h
    # Out (b, t, d) --> dOut  (b, t, d)
    b,  num_heads, t,  head_d = Q.shape
    dOut = dOut.reshape(b, t, num_heads, head_d)
    dOut = np.transpose(dOut, (0, 2, 1, 3))
    # dout  = (b,  num_heads,t,  head_d)
    dP = dOut @ np.transpose(V, (0, 1, 3, 2))
    dV = np.transpose(P, (0, 1, 3, 2)) @ dOut
    d = Q.shape[-1]
    print(d)
    dS = P * (dP - np.sum(dP * P, axis=-1, keepdims=True))
    dQ = (dS @ K) * (1. / np.sqrt(d))

    dK = (np.transpose(dS, (0, 1, 3, 2)) @ Q) * (1. / np.sqrt(d))

    dK = np.transpose(dK, (0, 2 , 1, 3))
    dQ = np.transpose(dQ, (0, 2 , 1, 3))
    dV = np.transpose(dV, (0, 2 , 1, 3))

    dK = np.reshape(dK, (b, t, head_d*num_heads))
    dQ = np.reshape(dQ, (b, t, head_d*num_heads))
    dV = np.reshape(dV, (b, t, head_d*num_heads))
    return dV, dK, dQ
def sanity_check_multi_head(R, Q, K, V, Eps):
    parameters_list = [Q, K, V]
    ans = []
    for parameters in parameters_list:
        b, r, c = parameters.shape
        dr_parameters = parameters.copy()
        for i in range (b):
          for j in range(r):
             for k in range(c):
                origin  = parameters[i][j][k]
                parameters[i][j][k] =  parameters[i][j][k] + Eps
                SoftMax, Out = attention(*parameters_list)
                L2 = np.sum(Out * R)
                parameters[i][j][k] = origin
                parameters[i][j][k] = parameters[i][j][k] - Eps
                SoftMax, Out = attention(*parameters_list)
                L1 = np.sum(Out * R)
                dr_parameters [i][j][k] = (L2 - L1) / (2*Eps)
                parameters[i][j][k] = origin
        ans.append(dr_parameters)
    return ans

def sanity_check_gradient_multihead(R, Q, K, V, num_heads, Eps):
    parameter_list = [Q, K, V]; ans = []
    for parameter in parameter_list:
        b, t,  d = parameter.shape
        dr_parameter = np.empty_like(parameter)
        for i in range(b):
            for j in range(t):
                for k in range(d):
                    origin = parameter[i][j][k]
                    parameter[i][j][k] = parameter[i][j][k] + Eps
                    Out, cash = multi_head_attention_forward(*parameter_list, num_heads)
                    L2 = np.sum(Out * R)
                    parameter[i][j][k] =  origin
                    parameter[i][j][k] = parameter[i][j][k] - Eps
                    Out, cash = multi_head_attention_forward(*parameter_list, num_heads)
                    L1 = np.sum(Out * R)
                    parameter[i][j][k] = origin
                    dr_parameter [i] [j] [k] = (L2 - L1)/(2*Eps)
        ans.append(dr_parameter)
    return  ans
def mask_attention_abckward(R, Q, K, V):
    d = Q.shape[-1]
    P = (Q @ np.transpose(K, (0, 2, 1)))* 1./np.sqrt(d)
    t = P.shape[-1]
    Mask = np.tril(np.ones((t, t), dtype=bool))
    P = np.where(Mask, P, -1e11)
    maxx = np.max(P, axis= -1, keepdims= True)
    EXP = np.exp(P-maxx)
    S = EXP/np.sum(EXP, axis=-1, keepdims= True)
    # print("S[0]:\n", S[0])
    #
    # print("V[0]:\n", V[0])


    Out = S @ V
    # print("Out[0]:\n", Out[0])

    dV = np.transpose(S, (0, 2, 1)) @ R

    dS =  R @ np.transpose(V, (0, 2, 1))
    dP = S * (dS - np.sum(dS*S, axis= -1, keepdims= True))
    dQ = (dP @ K)*1./np.sqrt(d)
    dK =  (np.transpose(dP, (0, 2, 1)) @ Q)*1./np.sqrt(d)
    return dQ, dK, dV

def sanity_check_gradient_mask(R, Q, K, V, eps):
    d = Q.shape[-1]
    def forward(Q, K, V):
        P = Q @ np.transpose(K, (0, 2, 1)) * (1./np.sqrt(d))
        dimm = P.shape[-1]
        mask = np.tril(np.ones((dimm, dimm ), dtype =bool))
        P = np.where(mask, P, -1e10)
        maxx_ = np.max(P, axis= -1,  keepdims= True)
        N = np.exp(P - maxx_)
        S = N/np.sum(N, axis= -1, keepdims=True)
        Out =  S @ V
        return  Out

    parameters_list = [Q, K, V]
    derivative_parameter = []
    for parameters in parameters_list:
        b,  t,  head_d = parameters.shape
        parameters_dr = np.empty_like(parameters)
        for i in range(b):
            for  j in range(t):
                for k in range(head_d):
                    origin = parameters[i][j][k]
                    parameters[i][j][k] += eps
                    xx =forward(*parameters_list)
                    L2 = np.sum(R*xx)
                    parameters[i][j][k] = origin
                    parameters[i][j][k] -= eps
                    xx = forward(*parameters_list)
                    L1  = np.sum(R*xx)
                    parameters_dr[i][j][k] = (L2-L1)/(2*eps)
                    parameters[i][j][k] = origin
        derivative_parameter.append( parameters_dr)
    dQ, dK, dV = derivative_parameter[:]
    return dQ, dK, dV

def padding_mask_forward_backward(R, Q, K, V, mask):
    # Q, K, V (b, t, head_d)
    d = Q.shape[-1]
    b = Q.shape[0]
    t = Q.shape[1]
    P = (Q @ np.transpose(K, (0, 2, 1))) * (1./np.sqrt(d))
    mask = mask.reshape(b, 1, t)
    P = np.where(mask, P, -1e10)
    max_ = np.max(P, axis= -1, keepdims= True)
    N =  np.exp(P-max_)
    # N = np.exp(P)
    S = N/np.sum(N, axis=-1, keepdims= True)
    out  = S @ V
    dV = np.transpose(S, (0, 2, 1)) @ R
    dS = R @ np.transpose(V,  (0, 2, 1))
    dP = S * (dS - np.sum(dS*S, axis= -1, keepdims=True))
    dQ = (dP @ K)* (1./np.sqrt(d))
    dK = (np.transpose(dP, (0, 2, 1)) @ Q) * (1./np.sqrt(d))
    return out, dQ, dK, dV

def padding_mask_gradient_sanity_check(R, Q, K, V, mask,  eps = 1e-5):
    def forward(Q, K, V, mask):
        d = Q.shape[-1]
        P =  (Q @ np.transpose(K, (0, 2, 1))) * (1./np.sqrt(d))
        # mask shape is (b, t)
        mask = mask[:, None, :]
        P = np.where(mask, P, -1e10)
        max_ = np.max(P, axis= -1, keepdims= True)

        N = np.exp(P- max_)
        S = N/np.sum(N, axis=-1, keepdims= True)
        out = S @ V
        return  out
    parameter_list = [Q, K, V]
    derivitave_parameters_list = []
    for parameter in parameter_list:
        dr_parameter = np.empty_like(parameter)
        b, r, c = Q.shape
        for i in range(b):
            for j in range(r):
                for k in range(c):
                    origin = parameter[i][j][k]
                    parameter[i][j][k] += eps
                    X2 = forward(*parameter_list, mask)
                    L2 = np.sum(R*X2)
                    parameter[i][j][k] = origin
                    parameter[i][j][k] -= eps
                    X1 = forward(*parameter_list, mask)
                    L1 = np.sum(R * X1)
                    dr_parameter [i][j][k] = (L2-L1)/(2*eps)
                    parameter[i][j][k] = origin
        derivitave_parameters_list.append(dr_parameter)
    dQ, dK, dV = derivitave_parameters_list
    return dQ, dK, dV
def causal_padded_mask_forward_backward(R, Q, K, V, mask, esps = 1e-5):
    def backward(R, Q, K, V, mask):
        d =  Q.shape[-1]
        P =  Q @ np.transpose(K, (0, 2, 1)) * (1./np.sqrt(d))
        t = P.shape[-1]
        mask = mask[:, None, :]
        P = np.where(mask, P, -1e10)
        mask_ = np.tril(np.ones((t, t), dtype= bool))
        P = np.where(mask_, P, -1e10)
        max_ = np.max(P, axis= -1, keepdims= True)
        N = np.exp(P-max_)
        S = N/np.sum(N, axis=-1, keepdims= True)
        out = S @ V
        dS = R @ np.transpose(V, (0, 2, 1))
        dV = np.transpose(S, (0, 2, 1)) @ R
        dP = S * (dS - np.sum(dS*S, axis= -1, keepdims= True))
        dQ = (dP @ K) * (1./np.sqrt(d))
        dK  = (np.transpose(dP, (0, 2, 1)) @ Q) * (1./np.sqrt(d))
        return dQ, dK, dV



    def forward(Q, K, V, mask):
        d = Q.shape[-1]
        P = Q @ np.transpose(K, (0, 2, 1)) * (1./np.sqrt(d))
        t = P.shape[-1]
        mask_casual = np.tril(np.ones((t,t), dtype =bool)) #
        P = np.where(mask_casual, P, -1e10)
        mask = mask[:, None, :]
        P = np.where(mask, P, -1e10)
        max_ = np.max(P, axis= -1, keepdims= True)
        N= np.exp(P - max_)
        S = N/np.sum(N, axis= -1, keepdims= True)
        out =  S @ V
        return out
    parameter_list = [Q, K, V]
    ans = []
    for parameter in parameter_list:
        b, r, c = parameter.shape
        dr_parameter = np.empty_like(parameter)
        for i in range(b):
            for j in range(r):
                for k in range(c):
                    origin = parameter [i][j][k]
                    parameter[i][j][k] += esps
                    X = forward(*parameter_list, mask)
                    L2 = np.sum(X * R)
                    parameter[i][j][k] = origin
                    parameter[i][j][k] -= esps
                    X = forward(*parameter_list, mask)
                    L1 = np.sum(X * R)
                    dr_parameter[i][j][k] = (L2-L1)/(2*esps)
                    parameter[i][j][k] = origin
        ans.append(dr_parameter)
    dQf, dKf, dVf = ans
    dQ, dK, dV = backward(R,Q, K, V, mask)
    out  = forward(Q, K, V,mask)
    return dQ, dK, dV, dQf, dKf, dVf, out







if __name__ == "__main__":
    # np.random.seed(0)
    # Q = np.zeros((1, 4, 2))
    # K = np.zeros((1, 4, 2))
    # V = np.random.randn(1, 4, 2)
    # att = mask_attention(Q, K, V)
    # print(f'shape of attention {att.shape}')
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2222
    # np.random.seed(0)
    # b, t, d = 2, 3, 4
    # Q = np.random.randn(b, t, d)
    # K = np.random.randn(b, t, d)
    # V = np.random.randn(b, t, d)
    # R = np.random.randn(b, t, d)  # upstream gradient
    # dV, dK, dQ = backprob_attention(R, Q, K, V)
    # ans = sanity_check(R, Q, K, V, Eps=1e-5)
    # dQf, dKf, dVf = ans
    # print(f'V is : {np.max(np.abs(dV-dVf))}')
    # print(f'Q is : {np.max(np.abs(dQ-dQf))}')
    # print(f'K is : {np.max(np.abs(dK-dKf))}')
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # np.random.seed(0)
    # b, t, d_model = 2, 4, 8
    # h = 2
    # Q = np.random.randn(b, t, d_model)
    # K = np.random.randn(b, t, d_model)
    # V = np.random.randn(b, t, d_model)
    # out = multi_head_attention_forward(Q, K, V, h)
    # print(out.shape)
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    #
    # np.random.seed(0)
    # b, t, d_model = 2, 3, 8
    # h = 2
    # Q = np.random.randn(b, t, d_model)
    # K = np.random.randn(b, t, d_model)
    # V = np.random.randn(b, t, d_model)
    # R = np.random.randn(b, t, d_model)
    # eps = 1e-5
    # Out, cash = multi_head_attention_forward(Q, K, V, h)
    # dV, dK, dQ = backward_multihead_attention(R, cash)
    # ans = sanity_check_gradient_multihead(R,Q, K, V, h, eps)
    #
    # print(f'Q: {np.max(ans[0]-dQ)}')
    # print(f'K: {np.max(ans[1]-dK)}')
    # print(f'V: {np.max(ans[2]-dV)}')
##################################################################
    # np.random.seed(0)
    # b, t, d = 1, 4, 2
    # Q = np.zeros((b, t, d))
    # K = np.zeros((b, t, d))
    # V = np.array([[
    #     [10.0, 0.0],  # token 0
    #     [0.0, 10.0],  # token 1
    #     [10.0, 10.0],  # token 2
    #     [-10.0, 5.0],  # token 3
    # ]])
    # R = np.ones((b, t, d))  # simple upstream
    # dQ, dK, dV = mask_attention_abckward(R, Q, K, V)
    # dQs, dKs, dVs =  sanity_check_gradient_mask(R, Q, K, V, 1e-4)
    # max_err_Q = np.max(np.abs(dQ - dQs))
    # max_err_K = np.max(np.abs(dK - dKs))
    # max_err_V = np.max(np.abs(dV - dVs))
    # print(max_err_Q, max_err_K, max_err_V)
    # ######################################################

    # np.random.seed(0)
    #
    # b, t, d = 2, 4, 3
    #
    # Q = np.zeros((b, t, d))
    # K = np.zeros((b, t, d))
    #
    # V = np.array([
    #     [[10., 0., 0.],
    #      [0., 10., 0.],
    #      [0., 0., 10.],
    #      [5., 5., 5.]],
    #
    #     [[1., 2., 3.],
    #      [4., 5., 6.],
    #      [7., 8., 9.],
    #      [10., 11., 12.]]
    # ])
    #
    # # mask: batch 0 has len=2 tokens, batch 1 has len=3 tokens
    # mask = np.array([
    #     [1, 1, 0, 0],
    #     [1, 1, 1, 0],
    # ], dtype=bool)
    #
    # # upstream gradient
    # R = np.ones((b, t, d))
    # out,  dQ, dK, dV = padding_mask_forward_backward(R, Q, K, V, mask)
    # print(out.shape, dQ.shape, dK.shape, dV.shape)
    # ################################################

    # np.random.seed(0)
    # Q = np.random.randn(2, 5, 4)
    # K = np.random.randn(2, 5, 4)
    # V = np.random.randn(2, 5, 4)
    # R = np.random.randn(2, 5, 4)
    # mask = np.array([
    #     [1,1,1,0,0],
    #     [1,1,1,1,0],
    # ], dtype=bool)
    #
    # out, dQ, dK, dV = padding_mask_forward_backward(R, Q, K, V, mask)
    # dQf, dKf, dVf = padding_mask_gradient_sanity_check(R, Q, K, V, mask, eps=1e-5)
    # print(f'max_err_Q {np.max(dQ-dQf)}')
    # print(f'max_err_K {np.max(dK-dKf)}')
    # print(f'max_err_V {np.max(dV-dVf)}')

    #######################################
    # import numpy as np
    #
    # np.random.seed(0)
    #
    # b, t, d = 2, 4, 3
    # Q = np.random.randn(b, t, d)
    # K = np.random.randn(b, t, d)
    # V = np.random.randn(b, t, d)
    # R = np.random.randn(b, t, d)
    #
    # mask = np.array([
    #     [1, 1, 1, 0],
    #     [1, 1, 0, 0],
    # ], dtype=bool)
    #
    # eps = 1e-5
    # dQ, dK, dV, dQf, dKf, dVf, out = causal_padded_mask_forward_backward(R, Q, K, V, mask, esps=1e-5)
    # print(f'max_err_Q {np.max(dQ-dQf)}')
    # print(f'max_err_K {np.max(dK-dKf)}')
    # print(f'max_err_V {np.max(dV-dVf)}')

    import numpy as np

    b, t, d = 2, 4, 3

    Q = np.zeros((b, t, d))
    K = np.zeros((b, t, d))

    V = np.array([
        # sample 0
        [[10., 0., 0.],
         [0., 10., 0.],
         [0., 0., 10.],
         [5., 5., 5.]],
        # sample 1
        [[1., 2., 3.],
         [4., 5., 6.],
         [7., 8., 9.],
         [-1., -2., -3.]]
    ])

    R = np.ones((b, t, d))  # upstream gradient for L = sum(out * R)

    mask = np.array([
        [1, 1, 1, 0],  # sample 0: token 3 is PAD (key position 3 masked)
        [1, 1, 0, 0],  # sample 1: tokens 2,3 are PAD
    ], dtype=bool)
    dQ, dK, dV, dQf, dKf, dVf, out = causal_padded_mask_forward_backward(R, Q, K, V, mask, esps=1e-5)
    print(out)




    

