
#include "BTC.h"

void BTC::EDBTC(const cv::Mat &src,cv::Mat &dst,const int BlockSize,const int Choose){

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

	double **image_reg;                       //�Ȧs
	image_reg = new double*[m];
	for(int i=0;i<m;i++)
		image_reg[i] = new double [n];

	//�v����J�μȦs
	//=============================================
	cv::Mat tsrc(src.rows,src.cols,CV_8UC1);
	for(int i=0 ; i<src.rows ; i++ ){
		for(int j=0 ; j<src.cols ; j++){
			image_in[i][j]=src.data[i*src.cols+j];
			image_reg[i][j]=src.data[i*src.cols+j];
		}
	}

	//�~�t�X���v��
	//=============================================
	double error_kernel1[3][3]={0,0,0,
		                        0,0,7,
	                            3,5,1};       //Floyd

	double error_kernel2[5][5]={0,0,0,0,0,
		                        0,0,0,0,0,
		                        0,0,0,7,5,
		                        3,5,7,5,3 ,
							    1,3,5,3,1};   //Jarvis
	double error_kernel3[5][5]={0,0,0,0,0,
		                        0,0,0,0,0,
		                        0,0,0,8,4,
		                        2,4,8,4,2,
							    1,2,4,2,1};   //Stucki

	//�p��Ѽ�(�D�P�B)
	//=============================================
	int y_start,x_start,y_end,x_end;                    //�]�w�B�̰϶����d��,����W�X���
	double a,b,mean,error,error_diffuse;                //a,b���q�Ƶ���,error���q�Ƶ��ŻP��Ƕ��Ȫ��~�t,error_diffuse���~�t�v�����`�M
	double total=BlockSize*BlockSize;                   //total���϶��j�p
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
			a=255;
			b=0;
			mean=0;
			for(int y=y_start ; y<y_end ; y++){
				for(int x=x_start ; x<x_end ; x++){
					mean=mean+image_in[y][x];
					if(image_in[y][x]<a)
						a=image_in[y][x];
					if(image_in[y][x]>b)
						b=image_in[y][x];
				}
			}
			mean=mean/total;
			for(int y=y_start ; y<y_end ; y++){
				for(int x=x_start ; x<x_end ; x++){
					if(image_reg[y][x]<mean)
						image_out[y][x]=a;
					else
						image_out[y][x]=b;
					error=image_reg[y][x]-image_out[y][x];
					if(Choose==1){
						for(int q=-1;q<=1;q++){
							for(int w=-1;w<=1;w++){
								if((y+q)>=0&&(x+w)>=0&&(y+q)<m&&(x+w)<n){
									error_diffuse=error*error_kernel1[q+1][w+1]/16;
									image_reg[y+q][x+w]=image_reg[y+q][x+w]+error_diffuse;
								}
							}
						}
					}
					else if(Choose==2){						
						for(int q=-2;q<=2;q++){
							for(int w=-2;w<=2;w++){
								if((y+q)>=0&&(x+w)>=0&&(y+q)<m&&(x+w)<n){
									error_diffuse=error*error_kernel2[q+2][w+2]/48;
									image_reg[y+q][x+w]=image_reg[y+q][x+w]+error_diffuse;
								}
							}
						}
					}
					else if(Choose==3){
						for(int q=-2;q<=2;q++){
							for(int w=-2;w<=2;w++){
								if((y+q)>=0&&(x+w)>=0&&(y+q)<m&&(x+w)<n){
									error_diffuse=error*error_kernel3[q+2][w+2]/42;
									image_reg[y+q][x+w]=image_reg[y+q][x+w]+error_diffuse;
								}
							}
						}
					}
				}					
			}
		}
	}

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
	delete	[]	image_reg;
	delete	[]	image_out;

}
