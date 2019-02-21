#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>

///////////////// api //////////////
// tensor.reshape(IntArrayRef shape) CONST
// tensor.mul(CONST Tensor &other) CONST
// tensor.slice(int64_t dim = 0, int64_t start = 0, int64_t end = 9223372036854775807, int64_t step = 1) CONST
//
///////////////////////////////////

at::Tensor index_natural_number(int max) // 这个函数产生自然数的下标
{
	at::Tensor index = at::_cast_Long(at::zeros({max}));
	index.print();
	for(int i=1; i<max; i++)
	{
		index[i] = index[i-1] + 1;
	}
	return index;
}

at::Tensor index_certain_number(int a)
{
	at::Tensor index = at::_cast_Long(at::zeros({1}));
	for(int i=0; i<a; i++)
	{
		index[0] = index[0] + 1;
	}
	return index;
}

std::vector<at::Tensor> trans_conv_forward(
		int in_channel,
		int out_channel,
		int kernel_size,
		int stride,
		int padding,
		int dilation,
		at::Tensor input,  // batch, in_channel, width, length
		at::Tensor depth_transform,  // batch, L, kernel_size**2
		at::Tensor weight){  // in_channel, kernel_size, kernel_size, out_channel
	// 循环遍历整张图，对每个block计算，给到结果
	int batch = input.size(0);
  int width = input.size(2) - (kernel_size-1) + 2*padding - 2*(dilation - 1);
  int length = input.size(3) - (kernel_size-1) + 2*padding - 2*(dilation - 1);
	auto output = torch::empty({width, length, batch, out_channel}).cuda();
  input = at::constant_pad_nd(input, {padding, padding, padding, padding}, 0);  //!!!!!!!!!!这里不知道padding是上下左右还是对称模式，要实验
	// auto input_a = input.accessor<float,4>();
	torch::Tensor de_tr, input_tmp, tmp;
	torch::Tensor input_block = torch::empty({3, 3, batch, in_channel}).cuda();
	for(int i=0; i<width; i++)
	{
		for(int j=0; j<length; j++)
		{
			for(int k=0; k<kernel_size; k++)
			{
				for(int l=0; l<kernel_size; l++)
				{
					input_block[k][l] = input.slice(2, i*stride+k*dilation, i*stride+k*dilation+1)
																	.slice(3, j*stride+l*dilation, j*stride+l*dilation+1)
																	.reshape({batch, in_channel});  // batch, in_channel, 1, 1, 遍历kernel
				}
			}
			de_tr = depth_transform.slice(1, i*length+j, i*length+j+1).reshape({batch, 1, kernel_size, kernel_size});  // batch, 1, 9 -> batch, 1, 3, 3
			input_tmp = input_block.permute({2, 3, 0, 1}).mul(de_tr);  // batch, in_channel, 3, 3
			output[i][j] = at::matmul(
				input_tmp.reshape({batch, in_channel*kernel_size*kernel_size}),
				weight.reshape({in_channel*kernel_size*kernel_size, out_channel})
			).reshape({batch, out_channel});
			// output.slice(2, i, i+1).slice(3, j, j+1) = tmp4;  // 原语句，思想是这个，函数不管用
		}
	}
	return {output.permute({2, 3, 0, 1})};
}

std::vector<at::Tensor> trans_conv_backward(
		int in_channel,
		int out_channel,
		int kernel_size,
		int stride,
		int padding,
		int dilation,
		at::Tensor input,  // batch, in_channel, width, length
		at::Tensor depth_transform,  // batch, L, kernel_size**2
		at::Tensor weight,	// in_channel, kernel_size, kernel_size, out_channel
		at::Tensor d_output){  // batch, out_channel, width', length'
	int batch = input.size(0);
	int in_width = input.size(2);
	int in_length = input.size(3);
	int width = input.size(2) - (kernel_size-1) + 2*padding - 2*(dilation - 1);
	int length = input.size(3) - (kernel_size-1) + 2*padding - 2*(dilation - 1);
	input = at::constant_pad_nd(input, {padding, padding, padding, padding}, 0);
	auto d_input = at::constant_pad_nd(at::zeros_like(input), {padding, padding, padding, padding}, 0).permute({2, 3, 0, 1});  //!!!!!!!!!!
	auto d_weight = at::zeros_like(weight).permute({1, 2, 3, 0});
	auto d_de_tr = at::zeros_like(depth_transform).permute({1, 2, 0});
	if(width != d_output.size(2)) {std::cout<<"wrong width!"<<std::endl;}
	if(length != d_output.size(3)) {std::cout<<"wrong length!"<<std::endl;}
	std::cout<<"input"<<std::endl;
	// d_input
	for(int i=0; i<width; i++)
	{
		for(int j=0; j<length; j++)
		{
			for(int k=0; k<kernel_size; k++)
			{
				for(int l=0; l<kernel_size; l++)
				{
					d_input[i*stride+k*dilation][j*stride+l*dilation]  // batch, in_channel
									+=
									at::mul(
										depth_transform.slice(1, i*length+j, i*length+j+1).
																		slice(2, k*kernel_size+l, k*kernel_size+l+1).reshape({batch, 1, 1}),  // batch, 1, 1
										weight.slice(1, k, k+1).slice(2, l, l+1).reshape({1, in_channel, out_channel})  // 1, in_channel, out_channel
									).mul(
										d_output.slice(2, i, i+1).slice(3, j, j+1).reshape({batch, 1, out_channel})  //batch, 1, out_channel
									).sum(2);
				}
			}
		}
	}
	d_input = d_input.permute({2, 3, 0, 1}).slice(2, 1, in_width+2-1).slice(3, 1, in_length+2-1);
	std::cout<<"weight"<<std::endl;
	// d_weight, in_channel,kernel_size, kernel_size, out_channel
	for(int i=0; i<width; i++)
	{
		for(int j=0; j<length; j++)
		{
			for(int k=0; k<kernel_size; k++)
			{
				for(int l=0; l<kernel_size; l++)
				{
					// input.slice(2, i*stride+k*dilation, i*stride+k*dilation+1).
					// 			slice(3, j*stride+l*dilation, j*stride+l*dilation+1).print();
					// depth_transform.slice(1, i*length+j, i*length+j+1).
					// 								slice(2, k*kernel_size+l, k*kernel_size+l+1).print();
					// d_output.slice(1, m, m+1).slice(2, i, i+1).slice(3, j, j+1).print();
					d_weight[k][l] // out_channel, in_channel
						+=
						at::mul(
							input.slice(2, i*stride+k*dilation, i*stride+k*dilation+1).
										slice(3, j*stride+l*dilation, j*stride+l*dilation+1).reshape({batch, 1, in_channel}),  // batch, 1, in_channel
							depth_transform.slice(1, i*length+j, i*length+j+1).
															slice(2, k*kernel_size+l, k*kernel_size+l+1).reshape({batch, 1, 1})  // batch, 1, 1
						).mul(
							d_output.slice(2, i, i+1).slice(3, j, j+1).reshape({batch, out_channel, 1 })  // batch, out_channel, 1, 1
						).sum(0).reshape({out_channel, in_channel}) / (float)batch;  // out_channel, in_channel
				}
			}
		}
	}
	d_weight = d_weight.permute({3, 0, 1, 2});
	std::cout<<"de_tr"<<std::endl;
	// d_de_tr
	for(int i=0; i<width; i++)
	{
		for(int j=0; j<length; j++)
		{
			for(int k=0; k<kernel_size; k++)
			{
				for(int l=0; l<kernel_size; l++)
				{
					// batch, in_channel, 1, 1 * in_channel, 1, 1, out_channel
					d_de_tr[i*length+j][k*kernel_size+l] // batch
									+=
									at::matmul(
										input.slice(2, i*stride+k*dilation, i*stride+k*dilation+1).
													slice(3, j*stride+l*dilation, j*stride+l*dilation+1).
													reshape({batch, in_channel}),  // batch, in_channel
										weight.slice(1, k, k+1).
													 slice(2, l, l+1).reshape({in_channel, out_channel})
									).mul(
										d_output.slice(2, i, i+1).slice(3, j, j+1).reshape({batch, out_channel})  // batch, out_channel
									).sum(1);
				}
			}
		}
	}
	d_de_tr = d_de_tr.permute({2, 0, 1});
	return {d_input, d_de_tr, d_weight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &trans_conv_forward, "trans_conv forward");
  m.def("backward", &trans_conv_backward, "trans_conv backward");
}
