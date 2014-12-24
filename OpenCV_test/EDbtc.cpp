
#include "BTC.h"

void BTC::EDBTC(const cv::Mat &src,cv::Mat &dst,const int BlockSize,const int Choose){

	//讀入影像的長寬
	//=============================================
	int m=src.rows, n=src.cols;	

	//動態記憶體配置所需的空間
	//=============================================
	int **image_in;
	image_in = new int*[m];                   //儲存輸入影像的灰階值
	for(int i=0;i<m;i++)
		image_in[i] = new int [n];

	int **image_out;                          //輸出處理後影像的灰階值
	image_out = new int*[m];
	for(int i=0;i<m;i++)
		image_out[i] = new int [n];

	double **image_reg;                       //暫存
	image_reg = new double*[m];
	for(int i=0;i<m;i++)
		image_reg[i] = new double [n];

	//影像輸入及暫存
	//=============================================
	cv::Mat tsrc(src.rows,src.cols,CV_8UC1);
	for(int i=0 ; i<src.rows ; i++ ){
		for(int j=0 ; j<src.cols ; j++){
			image_in[i][j]=src.data[i*src.cols+j];
			image_reg[i][j]=src.data[i*src.cols+j];
		}
	}

	//誤差擴散權重
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

	//計算參數(非同步)
	//=============================================
	int y_start,x_start,y_end,x_end;                    //設定處裡區塊的範圍,防止超出邊界
	double a,b,mean,error,error_diffuse;                //a,b為量化等級,error為量化等級與原灰階值的誤差,error_diffuse為誤差權重的總和
	double total=BlockSize*BlockSize;                   //total為區塊大小
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

	//影像輸出
	//=============================================
	cv::Mat tdst(src.rows,src.cols,CV_8UC1);
	for(int i=0 ; i<src.rows ; i++ )
		for(int j=0 ; j<src.cols ; j++)
			tdst.data[i*src.cols+j]=image_out[i][j];
	dst=tdst.clone();

	//釋放記憶體空間
	//=============================================
	delete	[]	image_in;
	delete	[]	image_reg;
	delete	[]	image_out;

}
