#include "BTC.h"

void BTC::BTC(const cv::Mat &src,cv::Mat &dst,const int BlockSize){

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

	//影像輸入及暫存
	//=============================================
	cv::Mat tsrc(src.rows,src.cols,CV_8UC1);
	for(int i=0 ; i<src.rows ; i++ )
		for(int j=0 ; j<src.cols ; j++)
			image_in[i][j]=src.data[i*src.cols+j];

	//計算參數(同步)
	//=============================================
	//int y_start,x_start,y_end,x_end;                    //設定處裡區塊的範圍,防止超出邊界
	//double a,b,q,mean,rms,sd;                           //a,b為量化等級,q計算大於等於mean的pixel數,mean是平均值,rms是平方平均值(方均根),sd為標準差
	//double total=BlockSize*BlockSize;                   //total為區塊大小
	//#pragma omp parallel num_threads(2)/*核心數設定*/ default(none) shared(m,n,image_in,image_out,BlockSize)/*共用參數宣告*/
	//{
        //#pragma omp for
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
	    		double rms=0;
	    		double total=BlockSize*BlockSize;
	    		for(int y=y_start ; y<y_end ; y++){
	    			for(int x=x_start ; x<x_end ; x++){
	    				mean=mean+image_in[y][x];
	    				rms=rms+image_in[y][x]*image_in[y][x];
	    			}
	    		}
	    		mean=mean/total;
	    		rms=rms/total;
	    		double sd=sqrt(rms-mean*mean);
	    		int q=0;
	    		for(int y=y_start ; y<y_end ; y++)
	    			for(int x=x_start ; x<x_end ; x++)
	    				if(image_in[y][x]>=mean)
	    					q++;
	    		double a=mean-sd*sqrt(q/(total-q));
	    		double b=mean+sd*sqrt((total-q)/q);
				if(a>255)
					a=255;
				if(a<0)
					a=0;
				if(b>255)
					b=255;
				if(b<0)
					b=0;
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
	delete	[]	image_out;
}