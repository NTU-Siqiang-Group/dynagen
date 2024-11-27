
class ProfilerConfig:
    def get_cache_size(self, batch_size, seq_len):
        raise NotImplementedError()
    
    def get_hidden_size(self, batch_size, seq_len):
        raise NotImplementedError()
    
    def get_weights(self):
        raise NotImplementedError()
    
    def get_htod_cost(self, size):
        return size * 1e-9
    
    def get_dtoh_cost(self, size):
        return size * 1e-9
    
    def get_compute_cache_gpu(self):
        return 1e-6
    
    def get_compute_cache_cpu(self):
        return 1e-5
    
    def get_compute_mlp_gpu(self):
        return 1e-7

class Llama1BConfig(ProfilerConfig):
    def get_weights(self):
        # attention_w_k_size = 2097152
        # attention_w_v_size = 2097152
        # attention_w_q_size = 8388608
        # attention_w_re_size = 64
        # attention_w_o_size = 8388608
        attention_weight_size = 20975680

        output_w_ln_size = 4096
        output_w_token_size = 525336576

        input_w_token_size = 525336576

        mlp_w_ln_size = 4096
        mlp_w_g_size = 33554432
        mlp_w_u_size = 33554432
        mlp_w_d_size = 33554432

        attention_size = attention_weight_size
        output_size = output_w_ln_size + output_w_token_size
        input_size = input_w_token_size
        mlp_size = mlp_w_ln_size + mlp_w_g_size + mlp_w_u_size + mlp_w_d_size

        weights = [input_size]
        for _ in range(16):
            weights.append(attention_size)
            weights.append(mlp_size)
        weights.append(output_size)
        return weights
    
    def get_cache_size(self, batch_size, seq_len):
        # input_dim: 2048
        return 2 * batch_size * seq_len * 2048 * 2
    
    def get_hidden_size(self, batch_size, seq_len):
        return batch_size * seq_len * 2048 * 2
