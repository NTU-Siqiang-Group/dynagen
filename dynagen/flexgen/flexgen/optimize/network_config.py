class ProfilerConfig:
    htod_cost = 2e-11
    dtoh_cost = 2e-11
    compute_cache_gpu = 5e-3
    compute_cache_cpu = 6e-3
    compute_mlp_gpu = 1e-4

    def get_cache_size(self, batch_size, seq_len):
        raise NotImplementedError()

    def get_hidden_size(self, batch_size, seq_len):
        raise NotImplementedError()

    def get_weights(self):
        raise NotImplementedError()

    def get_htod_cost(self, size):
        return size * self.htod_cost

    def get_dtoh_cost(self, size):
        return size * self.dtoh_cost

    def get_compute_cache_gpu(self):
        return self.compute_cache_gpu

    def get_compute_cache_cpu(self):
        return self.compute_cache_cpu

    def get_compute_mlp_gpu(self):
        return self.compute_mlp_gpu


class Llama1BConfig(ProfilerConfig):
    def get_weights(self):
        self.num_hidden_layers = 16

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

        self.attention_size = attention_weight_size
        output_size = output_w_ln_size + output_w_token_size
        input_size = input_w_token_size
        self.mlp_size = mlp_w_ln_size + mlp_w_g_size + mlp_w_u_size + mlp_w_d_size

        weights = [input_size]
        for _ in range(self.num_hidden_layers):
            weights.append(self.attention_size)
            weights.append(self.mlp_size)
        weights.append(output_size)
        return weights

    def get_mlp_size(self):
        self.get_weights()
        return self.mlp_size

    def get_attn_size(self):
        self.get_weights()
        return self.attention_size

    def get_cache_size(self, batch_size, seq_len):
        # input_dim: 2048
        return 2 * batch_size * seq_len * 2048 * 2

    def get_hidden_size(self, batch_size, seq_len):
        return batch_size * seq_len * 2048 * 2


class Llama13BConfig(ProfilerConfig):
    def get_weights(self):
        # 配置参数
        hidden_size = 5120
        intermediate_size = 13824
        vocab_size = 32000
        self.num_hidden_layers = 40
        dtype_bytes = 2  # float16每个参数2字节

        # Embedding
        input_w_token_size = vocab_size * hidden_size * dtype_bytes  # 32000*5120*2
        output_w_token_size = input_w_token_size

        # LayerNorm
        output_w_ln_size = hidden_size * dtype_bytes  # 5120*2
        mlp_w_ln_size = output_w_ln_size

        # MLP权重
        # W_g/W_u: [hidden_size, intermediate_size]
        # W_d: [intermediate_size, hidden_size]
        # 每个为70,656,000参数，*2字节=141,312,000 bytes
        # MLP共有3个大矩阵 W_g/W_u/W_d，共211,968,000参数 (3*70,656,000)，=423,936,000 bytes
        self.mlp_size = 423_936_000 + mlp_w_ln_size

        # Attention权重
        # Q, K, V, O: 每个[hidden_size, hidden_size] = 26,214,400参数
        # *2 bytes=52,428,800 bytes each
        # 四个合计209,715,200 bytes，加LN=+10,240
        self.attention_size = 209_715_200 + output_w_ln_size

        # Output层权重
        output_size = output_w_ln_size + output_w_token_size

        weights = [input_w_token_size]
        for _ in range(self.num_hidden_layers):
            weights.append(self.attention_size)
            weights.append(self.mlp_size)
        weights.append(output_size)

        return weights

    def get_mlp_size(self):
        self.get_weights()
        return self.mlp_size

    def get_attn_size(self):
        self.get_weights()
        return self.attention_size

    def get_cache_size(self, batch_size, seq_len):
        # 仿照之前的逻辑，KV缓存
        return 2 * batch_size * seq_len * 5120 * 2

    def get_hidden_size(self, batch_size, seq_len):
        return batch_size * seq_len * 5120 * 2


class Llama70BConfig(ProfilerConfig):
    def get_weights(self):
        hidden_size = 8192
        intermediate_size = 28672
        vocab_size = 32000
        self.num_hidden_layers = 80
        dtype_bytes = 2

        input_w_token_size = vocab_size * hidden_size * dtype_bytes
        output_w_token_size = input_w_token_size

        output_w_ln_size = hidden_size * dtype_bytes
        mlp_w_ln_size = output_w_ln_size

        mlp_w_g_size = intermediate_size * hidden_size * dtype_bytes
        mlp_w_u_size = intermediate_size * hidden_size * dtype_bytes
        mlp_w_d_size = hidden_size * intermediate_size * dtype_bytes

        self.mlp_size = mlp_w_g_size + mlp_w_u_size + mlp_w_d_size + mlp_w_ln_size

        attention_w_k_size = 1024 * hidden_size * dtype_bytes
        attention_w_v_size = 1024 * hidden_size * dtype_bytes
        attention_w_q_size = hidden_size * hidden_size * dtype_bytes
        attention_w_o_size = hidden_size * hidden_size * dtype_bytes
        attention_w_re_size = 64 * 4

        self.attention_size = attention_w_k_size + attention_w_v_size + attention_w_q_size + attention_w_o_size + attention_w_re_size + output_w_ln_size

        output_size = output_w_ln_size + output_w_token_size

        weights = [input_w_token_size]
        for _ in range(self.num_hidden_layers):
            weights.append(self.attention_size)
            weights.append(self.mlp_size)
        weights.append(output_size)

        return weights
    
    def get_mlp_size(self):
        self.get_weights()
        return self.mlp_size
    
    def get_attn_size(self):
        self.get_weights()
        return self.attention_size
    
    def get_cache_size(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * 8192 * 2
    
    def get_hidden_size(self, batch_size, seq_len):
        return batch_size * seq_len * 8192 * 2

class Opt13BConfig(ProfilerConfig):
    def get_weights(self):
        hidden_size = 5120
        intermediate_size = 13824
        vocab_size = 32000
        self.num_hidden_layers = 40
        dtype_bytes = 2

        input_w_token_size = vocab_size * hidden_size * dtype_bytes
        output_w_token_size = input_w_token_size

        output_w_ln_size = hidden_size * dtype_bytes
        mlp_w_ln_size = output_w_ln_size

        mlp_w_g_size = intermediate_size * hidden_size * dtype_bytes
        mlp_w_u_size = intermediate_size * hidden_size * dtype_bytes
        mlp_w_d_size = hidden_size * intermediate_size * dtype_bytes

        self.mlp_size = mlp_w_g_size + mlp_w_u_size + mlp_w_d_size + mlp_w_ln_size

        attention_w_k_size = hidden_size * hidden_size * dtype_bytes
        attention_w_v_size = hidden_size * hidden_size * dtype_bytes
        attention_w_q_size = hidden_size * hidden_size * dtype_bytes
        attention_w_re_size = 64 * dtype_bytes
        attention_w_o_size = hidden_size * hidden_size * dtype_bytes

        self.attention_size = attention_w_k_size + attention_w_v_size + attention_w_q_size + attention_w_re_size + attention_w_o_size + output_w_ln_size

        output_size = output_w_ln_size + output_w_token_size

        weights = [input_w_token_size]
        for _ in range(self.num_hidden_layers):
            weights.append(self.attention_size)
            weights.append(self.mlp_size)
        weights.append(output_size)

        return weights

    def get_mlp_size(self):
        self.get_weights()
        return self.mlp_size

    def get_attn_size(self):
        self.get_weights()
        return self.attention_size

    def get_cache_size(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * 5120 * 2

    def get_hidden_size(self, batch_size, seq_len):
        return batch_size * seq_len * 5120 * 2
