#include "BTC.h"

void BTC::AMBTC(const cv::Mat &src,cv::Mat &dst,const int BlockSize){

	//Ū�J�v�������e
	//=============================================
	int m=src.rows, n=src.cols;	

	//�ʺA�O����t�m�һݪ��Ŷ�
	//=============================================
	int **image_in;
	image_in = new int*[m];                   //�x�s��J�v�����Ƕ���
	for(int i=0;i<m;i++)
		image_in[i] = new int [n];

	int **image_out;                          //��X�B�z��v�����Ƕ���
	image_out = new int*[m];
	for(int i=0;i<m;i++)
		image_out[i] = new int [n];

	//�v����J�μȦs
	//=============================================
	cv::Mat tsrc(src.rows,src.cols,CV_8UC1);
	for(int i=0 ; i<src.rows ; i++ )
		for(int j=0 ; j<src.cols ; j++)
			image_in[i][j]=src.data[i*src.cols+j];

	//�p��Ѽ�(�P�B)
	//=============================================
	//int y_start,x_start,y_end,x_end;                    //�]�w�B�̰϶����d��,����W�X���
	//double a,b,p,q,mean;                                //a,b���q�Ƶ���,p�p��p��mean��pixel��,q�p��j�󵥩�mean��pixel��
	//double total=BlockSize*BlockSize;                   //total���϶��j�p
	//#pragma omp parallel num_threads(2)/*�֤߼Ƴ]�w*/ default(none) shared(m,n,image_in,image_out,BlockSize)/*�@�ΰѼƫŧi*/
	//{
    //    #pragma omp for
		for(int Y=0 ; Y<(m/BlockSize) ; Y++){
			for(int X=0 ; X<(n/BlockSize) ; X++){
				int y_start=Y*BlockSize;
				int x_start=X*BlockSize;
				int y_end=(Y+1)*BlockSize;
				if(y_end>m)
					y_end=m;
				int x_end=(X+1)*BlockSize;
				if(x_end>n)
					x_end=n;
				double mean=0;
				double total=BlockSize*BlockSize;
				for(int y=y_start ; y<y_end ; y++)
					for(int x=x_start ; x<x_end ; x++)
						mean=mean+image_in[y][x];
				mean=mean/total;
				double a=0;
				double b=0;
				double p=0;
				double q=0;
				for(int y=y_start ; y<y_end ; y++){
					for(int x=x_start ; x<x_end ; x++){
						if(image_in[y][x]<mean){
							p++;
							a=a+image_in[y][x];
						}
						else{
							q++;
							b=b+image_in[y][x];
						}
					}
				}
				a=a/p;
				b=b/q;
				for(int y=y_start ; y<y_end ; y++){
					for(int x=x_start ; x<x_end ; x++){
						if(image_in[y][x]>=mean)
							image_out[y][x]=b;
						else
							image_out[y][x]=a;
					}
				}
			}
		}
	//}

	//�v����X
	//=============================================
	cv::Mat tdst(src.rows,src.cols,CV_8UC1);
	for(int i=0 ; i<src.rows ; i++ )
		for(int j=0 ; j<src.cols ; j++)
			tdst.data[i*src.cols+j]=image_out[i][j];
	dst=tdst.clone();

	//����O����Ŷ�
	//=============================================
	delete	[]	image_in;
	delete	[]	image_out;

}