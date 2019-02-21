#include <time.h>
#include <iostream>
// #include <torch/torch.h>
#include <ATen/ATen.h>
// using namespace std;

// int main()
// {
// 	clock_t t1 = clock();
// 	int a;
// 	for(int i=0; i<480; i++)
// 	{
// 		int a;
// 		for(int j=0; j<640; j++)
// 		{
// 			a = i*j;
// 		}
// 	}
// 	cout << (clock() - t1) * 1.0 / CLOCKS_PER_SEC << endl;
// }

// at::Tensor b = at::randn({2, 2, 3});
// cout << b.size(0) << endl;
at::Tensor index_certain_number(int a)
{
	at::Tensor index = at::_cast_Long(at::zeros({1}));
	for(int i=0; i<a; i++)
	{
		index[0] = index[0] + 1;
	}
	return index;
}
int batch = 2;
int out_channel = 64;
int width = 480;
int length = 640;
auto output = at::zeros({batch, out_channel, width, length});
at::Tensor index3 = index_certain_number(i);
at::Tensor index4 = index_certain_number(j);
auto tmp4 = at::ones({batch, out_channel, 1, 1})
output.index_put({NULL, NULL, index3, index4}, tmp4);
output.print()
