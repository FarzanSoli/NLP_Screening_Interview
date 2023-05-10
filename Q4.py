"""
Created on Sat May  9 13:14:08 2023
@author: Soleymanif
"""
"""
    t:      time-step
    s:      increment of labels
    l's:    current_padded_char
"""
import numpy as np
# ====================================== # 
class Alpha_Beta_calculation():
    def __init__(self, gt_label, outputs):
        # gt_label == ground-truth label -> [l] from the paper
        self.outputs = outputs
        self.padded_gt_label = '-%s-' % '-'.join(gt_label)  # [l'] from the paper. 
        self.num_time_steps = outputs.shape[0]
        self.padded_gt_label_len = len(self.padded_gt_label)
        self.last_padded_ind = self.padded_gt_label_len - 1
        self.blank_char = Test_CTC().blank_char
        # Alpha (Forward Variable):
        self.Alpha_ = np.zeros((self.num_time_steps, self.padded_gt_label_len))
        # Beta (Backward Variable):
        self.Beta_ = np.zeros((self.num_time_steps, self.padded_gt_label_len))
    # Using dynamic programming to fill tables of size (T, |l'|) for alpha, beta.
    # --------------------------------- #
    def Forward_Alpha(self, t, s):

        if s < 0 or s >= len(self.padded_gt_label):
            return 0
        current_padded_char = self.padded_gt_label[s]
        current_padded_label_score = self.outputs[t, Test_CTC().alphabet_dict[current_padded_char]]
        if t == 0:
            if s == 0:
                return self.outputs[0, Test_CTC().blank_ind]
            elif s == 1:
                return current_padded_label_score
            else:
                return 0
        # Eq.(6, 7) from the paper. No need to call alpha for previous time-steps, because it was already calculated
        alpha_bar_t_s = self.Alpha_[t - 1, s] + (self.Alpha_[t - 1, s - 1] if s-1 >= 0 else 0)
        if current_padded_char == self.blank_char or (s >= 2 and self.padded_gt_label[s-2] == current_padded_char):
            return alpha_bar_t_s * current_padded_label_score
        else:
            return (alpha_bar_t_s + (self.Alpha_[t - 1, s - 2] if s - 2 >= 0 else 0)) * current_padded_label_score
    # --------------------------------- #
    def Backward_Beta(self, t, s):
        if s < 0 or s >= len(self.padded_gt_label):
            return 0
        current_padded_char = self.padded_gt_label[s]
        current_padded_label_score = self.outputs[t, Test_CTC().alphabet_dict[current_padded_char]]
        final_time_step = self.outputs.shape[0] - 1
        if t == final_time_step:
            if s == self.last_padded_ind:
                return self.outputs[final_time_step, Test_CTC().blank_ind]
            elif s == self.last_padded_ind - 1:
                return current_padded_label_score
            else:
                return 0
        # Eq.(10, 11) from the paper. No need to call beta for previous time-steps, because it was already calculated
        beta_bar_t_s = self.Beta_[t + 1, s] + (self.Beta_[t + 1, s + 1] if s + 1 <= self.last_padded_ind else 0)
        if current_padded_char == self.blank_char or \
                (s + 2 <= self.last_padded_ind and self.padded_gt_label[s+2] == current_padded_char):
            return beta_bar_t_s * current_padded_label_score
        else:
            return (beta_bar_t_s +
                    (self.Beta_[t + 1, s + 2] if s + 2 <= self.last_padded_ind else 0)) * current_padded_label_score
    # --------------------------------- #  
    def Alpha_Beta_calc(self):
        for t in range(0, self.num_time_steps):
            for s in range(0, self.padded_gt_label_len):
                self.Alpha_[t, s] = self.Forward_Alpha(t, s)
        # --------------------------------- #
        for t in range(self.num_time_steps - 1, -1, -1):
            for s in range(self.padded_gt_label_len - 1, -1, -1):
                self.Beta_[t, s] = self.Backward_Beta(t, s)
                
        return self.Alpha_, self.Beta_
# ====================================== # 
def test_forward_backward(outputs):
    label = ''.join([Test_CTC().test_alphabet[t % len(Test_CTC().test_alphabet)] for t in range(outputs.shape[0])])
    alpha_calc, beta_calc = Alpha_Beta_calculation(label, outputs).Alpha_Beta_calc()
    final_time_step_ind = outputs.shape[0] - 1
    padded_label_length = 2 * len(label) + 1
    label_score_manually = np.prod([outputs[t, Test_CTC().alphabet_dict[label[t]]] for t in range(outputs.shape[0])])
    # Eq. (8) from the paper
    score_last = alpha_calc[final_time_step_ind, padded_label_length - 1]
    score_before_last = alpha_calc[final_time_step_ind, padded_label_length - 2]
    label_score_by_alpha = score_last + score_before_last
    # similar to Eq. (8), P(label|outputs) is the sum of the probabilities of l' with and without the
    # first "blank" at time-step 0:
    score_first = beta_calc[0, 0]
    score_second = beta_calc[0, 1]
    label_score_by_beta = score_first + score_second
    print('> Forward-Backward Tests')
    print('\t CTC')
    print(outputs.T)
    print('\n\t GT Label: \'%s\'' % label)
    print('\n\t Tests')
    # Eq. (14) from the paper
    padded_label = '-%s-' % '-'.join(label)
    for t in range(outputs.shape[0]):
        prob_sum = 0.
        for s in range(padded_label_length):
            padded_char_s = padded_label[s]
            score_s_t = outputs[t, Test_CTC().alphabet_dict[padded_char_s]]
            alpha_beta_prod_t_s = alpha_calc[t, s] * beta_calc[t, s]
            prob_sum += alpha_beta_prod_t_s / score_s_t
    print('\n <<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>> \n')
    return prob_sum
# ====================================== # 
def calculate_gradients_CTC(outputs, gt_label):
    assert outputs.shape[0] >= len(gt_label)
    alpha_calc, beta_calc = Alpha_Beta_calculation(gt_label, outputs).Alpha_Beta_calc()
    padded_gt_label = '-%s-' % '-'.join(gt_label)
    gradients = np.zeros_like(outputs)
    score_last = alpha_calc[outputs.shape[0] - 1, len(padded_gt_label) - 1]
    score_before_last = alpha_calc[outputs.shape[0] - 1, len(padded_gt_label) - 2]
    p_l_given_ctc = score_last + score_before_last
    for t in range(outputs.shape[0]):
        for k in range(outputs.shape[1]):
            # Eq.(15):
            d_p_d_ytk = 0
            lab_lk = np.nonzero(
                list(map(lambda x: 1 if Test_CTC().index_dict[k] in x else 0, padded_gt_label)))[0]
            for s in lab_lk:
                d_p_d_ytk += alpha_calc[t, s] * beta_calc[t, s]

            d_p_d_ytk /= (outputs[t, k] ** 2)
            d_lnp_d_ytk = (1. / p_l_given_ctc) * d_p_d_ytk
            gradients[t, k] = d_lnp_d_ytk
    return gradients
# ====================================== # 
class Test_CTC:
    def __init__(self):
        super().__init__()
        self.blank_char = '-'
        self.test_alphabet = ['a', 'b', 'c', 'k']
        # given character, find their index
        self.alphabet_dict = {ch: ind for ind, ch in enumerate(self.test_alphabet + [self.blank_char])}
        # given index, find the character
        self.index_dict = {ind: ch for ind, ch in enumerate(self.test_alphabet + [self.blank_char])}
        self.blank_ind = self.alphabet_dict[self.blank_char]
    # --------------------------------- #
    def grads_print(self):
        random_ctc_matrix = np.random.rand(len(self.test_alphabet), len(self.alphabet_dict))
        gt_label = 'back'
        print('> Calculating gradients of the CTC matrix')
        ctc_matrix_grad = calculate_gradients_CTC(random_ctc_matrix, gt_label)
        print('\t CTC')
        print(random_ctc_matrix.T)
        print('\n\t GT Label: \'%s\'' % gt_label)
        print('\n\t CTC Gradients')
        print(ctc_matrix_grad.T)
# ====================================== # 
if __name__ == '__main__':
    random_ctc_matrix = np.random.rand(len(Test_CTC().test_alphabet), len(Test_CTC().alphabet_dict))
    prob_sum = test_forward_backward(random_ctc_matrix)
    Test_CTC().grads_print()


