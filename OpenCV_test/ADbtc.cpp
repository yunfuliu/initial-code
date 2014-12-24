#include "BTC.h"
using namespace std;

void cal_Parameter(int y_start, int y_end, int x_start, int x_end, int block_size,const cv::Mat &src, int *local_a, int *local_b, double *local_mean, double *local_sd){
	int a=255,b=0,number;
	double mean=0,u=0,sd=0;

	if(block_size==16)
		number=3;
	else if(block_size==8)
		number=2;
	else if(block_size==4)
		number=1;
	else if(block_size==2)
		number=0;
	for(int y=y_start ; y<y_end ; y++){
		for(int x=x_start ; x<x_end ; x++){
			mean=mean+src.data[y*src.cols+x];
			u=u+src.data[y*src.cols+x]*src.data[y*src.cols+x];
			if(src.data[y*src.cols+x]<a)
				a=src.data[y*src.cols+x];
			if(src.data[y*src.cols+x]>b)
				b=src.data[y*src.cols+x];
		}
	}
	*local_a=a;
	*local_b=b;
	mean=mean/block_size/block_size;
	*local_mean=mean;
	u=u/block_size/block_size;
	*local_sd=sqrt(u-mean*mean);
}
int arrange_CM(int row/*影像長度(m)*/, int col/*影像寬度(n)*/, int Y, int X, int y_start, int y_end, int x_start, int x_end, int block_size,vector<vector<vector<int> > > &class_matrix, vector<vector<vector<vector<int> > > > &class_matrix_rotate){

	int condition=1,Same_CM=0,Same_PreocessOder=0;        //condition判斷條件, Same_CM計算是否CM相同, Same_PreocessOder計算是否處理順序相同
	int change_small=100,final1=8,final2=8,overloop=0;    //change_small符合的條件數量(越小越符合), final1.2現在CM的旋轉方式, overloop計算重複次數避免無限迴圈
	int count=0,number;

	if(block_size==16)
		number=3;
	else if(block_size==8)
		number=2;
	else if(block_size==4)
		number=1;
	else if(block_size==2)
		number=0;

	while(condition!=0){					
		int rand_choice=rand()%8;                  //隨機挑選一個旋轉後的CM
		for(int y=y_start ; y<y_end ; y++)
			for(int x=x_start ; x<x_end ; x++)
				class_matrix[number][y][x]=class_matrix_rotate[number][rand_choice][y%block_size][x%block_size];	
		Same_CM=0;
		Same_PreocessOder=0;
		if(X>=1){ //判斷是否與左方區塊相同
			count=0;
			for(int y=y_start ; y<y_end ; y++)
				for(int x=x_start ; x<x_end ; x++)
					if( class_matrix[number][y][x]==class_matrix[number][y][x-block_size] )
						count++;
			if(count==block_size*block_size)
				Same_CM++;
		}
		if(Y>=1){ //判斷是否與上方區塊相同
			count=0;
			for(int y=y_start ; y<y_end ; y++)
				for(int x=x_start ; x<x_end ; x++)
					if( class_matrix[number][y][x]==class_matrix[number][y-block_size][x] )
						count++;
			if(count==block_size*block_size)
				Same_CM++;
		}
		
		if(X>=1){ //判斷現在CM的左方處理順序是否會與周遭相同
			for(int y=y_start ; y<y_end ; y++){
				int x=x_start;
				if(y-1>=0){
					if(class_matrix[number][y][x]==class_matrix[number][y-1][x-1])
						Same_PreocessOder++;
				}
				if(y+1<row){
					if(class_matrix[number][y][x]==class_matrix[number][y+1][x-1])
						Same_PreocessOder++;
				}
				if(class_matrix[number][y][x]==class_matrix[number][y][x-1])
					Same_PreocessOder++;
			}
		}
		if(Y>=1){ //判斷現在CM的上方處理順序是否會與周遭相同
			for(int x=x_start ; x<x_end ; x++){
				int y=y_start;
				if(x-1>=0){
					if(class_matrix[number][y][x]==class_matrix[number][y-1][x-1])
						Same_PreocessOder++;
				}
				if(x+1<col){
					if(class_matrix[number][y][x]==class_matrix[number][y-1][x+1])
						Same_PreocessOder++;
				}
				if(class_matrix[number][y][x]==class_matrix[number][y-1][x])
					Same_PreocessOder++;
			}
		}
		
		if(Same_CM==0 && Same_PreocessOder==0){  //條件達成則跳出迴圈
			condition=0;
			return rand_choice;
		}
		else{
			if(Same_CM==0 && change_small>Same_PreocessOder){ //條件未達成,儲存達成最多條件的CM
				change_small=Same_PreocessOder;
				final1=rand_choice;
			}
			if(change_small>Same_PreocessOder)
				final2=rand_choice;
		}
		
		if(overloop>8){                        //當選擇不出符合條件的CM時,選擇符合最多條件的CM,但仍要避免與左方及上方CM相同
			if(final1!=0 && final1!=1 && final1!=2 && final1!=3 && final1!=4 && final1!=5 && final1!=6 && final1!=7)
				rand_choice=final2;
			else
				rand_choice=final1;
			for(int y=y_start ; y<y_end ; y++)
				for(int x=x_start ; x<x_end ; x++)
					class_matrix[number][y][x]=class_matrix_rotate[number][rand_choice][y%block_size][x%block_size];
			condition=0;
			return rand_choice;
		}
		overloop++;
	}
}

void BTC::ADBTC(const cv::Mat &src,cv::Mat &dst,const int ImageQuality){

	//計算時間-起始 //(顯示結果用"可刪除")
	clock_t start_time, mid1_time, mid2_time, mid_end_time, final_end_time;
	double total_time = 0;
	start_time = clock(); //mircosecond

	//讀入影像的長寬
	//=============================================
	int m=src.rows, n=src.cols;

	//CM, DM
	//=============================================
	int class_matrix2[2][2]={ 2,1,
		                      0,3 };
	double diffused_matrix2[3][3]={ 1.808594,   1,	1.808594,
                                           1,   0, 	1,
								    1.808594,   1, 	1.808594  }; 
	
	int class_matrix4[4][4]={ 3, 9, 10, 2 , 
                              6, 8, 12, 13, 
                              0, 1, 14, 15, 
							  4, 5, 7 , 11 };
	double diffused_matrix4[3][3]={ 0.209961,   1,	0.209961,
                                           1,   0, 	1,
								    0.209961,   1, 	0.209961 };
	
	int class_matrix8[8][8]={  2,    32,     5,    18,    35,    38,    11,    12,   
                              24,    62,    61,    50,    47,    45,    23,    16,   
                              15,    10,    55,    56,    37,    63,    13,    20,   
                              28,    41,    54,    33,    57,    58,    36,    29,   
                              52,    53,    46,    42,     8,    51,    40,    30,   
                              60,    48,    43,     9,    31,    26,    49,    44,   
                               4,    14,    21,    25,    27,    34,    39,    59,   
						 	   6,     1,    19,    17,    22,     3,     7,     0 };
	double diffused_matrix8[3][3]={ 0.2564,	   1,	0.2564,
                                         1,	   0, 	1,
								    0.2564,	   1, 	0.2564 };

	int class_matrix16[16][16]={   164,   144,   114,    63,    39,    42,    47,    44,     6,   151,    95,   162,   165,    76,   246,   203,   
                                   244,    13,    21,    50,    60,    57,    82,    89,   104,   145,   158,    87,   177,   196,   197,   206,   
                                     5,    34,    27,     2,    58,    70,    98,    99,    15,   157,   130,   186,   137,    97,   208,    79,   
                                    23,    25,    41,    53,    67,    51,   119,   112,   142,   159,   176,   201,   222,   223,   225,   229,   
                                    22,    16,    48,    20,    94,   226,   103,   133,   111,   173,   166,   221,   184,   170,   227,     4,   
                                    31,   100,    66,    35,   106,   117,   155,   138,    73,   182,   193,   224,   232,   238,   187,    32,   
                                     3,   120,   102,   105,   123,   135,    96,   179,   198,   202,   220,   230,     7,   235,    17,    43,   
                                   153,   122,    83,   109,   160,   118,   178,   183,   204,   194,   150,   231,   255,    11,    56,    49,   
                                    40,   131,   116,   147,   169,   175,   113,   213,   218,   242,   243,   248,   247,    33,    52,    68,   
                                   108,   136,   140,    84,   185,   128,   214,   217,   233,   152,   249,    28,   181,    45,    74,    71,   
                                   110,   141,    88,    75,   192,   205,   195,   234,   127,   237,   253,    38,    65,    77,    72,   115,   
                                   146,   148,   161,   174,   124,   211,   168,   240,   251,   252,   254,    55,    64,    12,   167,   125,   
                                   149,   180,   156,   191,    81,   216,   236,   245,    26,     0,   171,    46,    92,   101,   143,   132,   
                                   189,   188,   107,   209,   210,   228,   250,    18,    37,    59,    69,    91,   134,   139,    85,   163,   
                                     1,   199,   241,   212,    93,     8,    30,    36,    61,    62,    90,    86,   239,   129,   154,   172,   
								   200,   207,   219,     9,    10,    24,    14,    54,    29,    78,    80,   121,   126,    19,   215,   190};
	double diffused_matrix16[3][3]={ 0.3165,   1,	0.3165,
                                          1,   0,	1,
									 0.3165,   1,	0.3165 };

	//Beta
	//=============================================
	vector<double> beta2(128,0);
	vector<double> beta4(128,0);
	vector<double> beta8(128,0);
	vector<double> beta16(128,0);
	int judge16=0,judge8=0,judge4=0,judge2=0; //判斷當曲線小於0時 之後的曲線值皆為0(因為這公式後端會往上跑回正值)
	for(int i=0; i<128; i++){
		beta16[i] = -1.5716E-12*i*i*i*i*i*i + 5.4796E-10*i*i*i*i*i - 7.4589E-08*i*i*i*i + 5.4849E-06*i*i*i - 2.5452E-04*i*i + 3.2634E-03*i + 3.4302E-01;
		if(beta16[i]<=0)
			judge16=1;
		if(judge16==1)
			beta16[i]=0;
		
		beta8[i] = -7.0371E-13*i*i*i*i*i*i + 2.2610E-10*i*i*i*i*i - 2.9332E-08*i*i*i*i + 2.3608E-06*i*i*i - 1.4796E-04*i*i + 2.5414E-03*i + 3.2729E-01;
		if(beta8[i]<=0)
			judge8=1;
		if(judge8==1)
			beta8[i]=0;
		
		beta4[i] = -1.4973E-12*i*i*i*i*i*i + 6.0742E-10*i*i*i*i*i - 9.1932E-08*i*i*i*i + 6.4693E-06*i*i*i - 2.3414E-04*i*i + 3.0048E-03*i + 2.6201E-01;

		if(beta4[i]<=0)
			judge4=1;
		if(judge4==1)
			beta4[i]=0;
		
		beta2[i] = -1.5251E-13*i*i*i*i*i*i + 5.0776E-11*i*i*i*i*i - 1.1594E-09*i*i*i*i - 6.4311E-07*i*i*i + 2.9603E-05*i*i - 1.0307E-03*i + 2.3843E-01;
		if(beta2[i]<=0)
			judge2=1;
		if(judge2==1)
			beta2[i]=0;
	}

	//配置所需的vector
	//=============================================
	vector<vector<double>> image_reg(m,vector<double>(n,0));  //暫存輸入影像的灰階值
	vector<vector<int>> image_out(m,vector<int>(n,0));  //輸出處理後影像的灰階值
	vector<vector<vector<int> > > class_matrix(4, vector<vector<int> >(512, vector<int>(512))); //儲存預先排好的4種不同大小CM

	//0729新增
	//存不同大小CM旋轉方式(0~7) 若沒有則為8
	vector<vector<int>> CM_reg2(m/2,vector<int>(n/2,0)); //2
	vector<vector<int>> CM_reg4(m/4,vector<int>(n/4,0)); //4
	vector<vector<int>> CM_reg8(m/8,vector<int>(n/8,0)); //8
	vector<vector<int>> CM_reg16(m/16,vector<int>(n/16,0)); //16

	//影像暫存
	//=============================================
	for(int i=0 ; i<src.rows ; i++ )
		for(int j=0 ; j<src.cols ; j++)
			image_reg[i][j]=src.data[i*src.cols+j];
			
	//CM旋轉處理
	//=============================================
	vector<vector<int>> cmr(16,vector<int>(16,0));
	//第一個[]存size(ex:0表示CM為2*2 1表示4*4以此類推) 第二個[]存第幾種旋轉 第三.四個[][]存CM的數值
	vector<vector<vector<vector<int> > > > class_matrix_rotate(4, vector<vector<vector<int> > >(8, vector<vector<int> >(16, vector<int>(16,0))));
	
	for(int rotate=0;rotate<4;rotate++)	{
		int n_rotate;
		if(rotate==0){
			n_rotate=2;
			for(int i=0;i<n_rotate;i++)
				for(int j=0;j<n_rotate;j++)
					cmr[i][j]=class_matrix2[i][j];
		}
		else if(rotate==1){
			n_rotate=4;
			for(int i=0;i<n_rotate;i++)
				for(int j=0;j<n_rotate;j++)
					cmr[i][j]=class_matrix4[i][j];
		}
		else if(rotate==2){
			n_rotate=8;
			for(int i=0;i<n_rotate;i++)
				for(int j=0;j<n_rotate;j++)
					cmr[i][j]=class_matrix8[i][j];
		}
		else if(rotate==3){
			n_rotate=16;
			for(int i=0;i<n_rotate;i++)
				for(int j=0;j<n_rotate;j++)
					cmr[i][j]=class_matrix16[i][j];
		}		
		for(int i=0;i<n_rotate;i++)
			for(int j=0;j<n_rotate;j++)
				class_matrix_rotate[rotate][0][i][j]=cmr[i][j]; //0存還沒旋轉的
		for(int i=0;i<n_rotate;i++)
			for(int j=0;j<n_rotate;j++)
				class_matrix_rotate[rotate][1][i][j]=class_matrix_rotate[rotate][0][(n_rotate-1)-j][i]; //1存0右轉一次的
		for(int i=0;i<n_rotate;i++)
			for(int j=0;j<n_rotate;j++)
				class_matrix_rotate[rotate][2][i][j]=class_matrix_rotate[rotate][1][(n_rotate-1)-j][i]; //2存0右轉兩次的
		for(int i=0;i<n_rotate;i++)
			for(int j=0;j<n_rotate;j++)
				class_matrix_rotate[rotate][3][i][j]=class_matrix_rotate[rotate][2][(n_rotate-1)-j][i]; //3存0右轉三次的
		for(int i=0;i<n_rotate;i++)
			for(int j=0;j<n_rotate;j++)
				class_matrix_rotate[rotate][4][i][j]=class_matrix_rotate[rotate][0][j][i]; //4存0的transpose
		for(int i=0;i<n_rotate;i++)
			for(int j=0;j<n_rotate;j++)
				class_matrix_rotate[rotate][5][i][j]=class_matrix_rotate[rotate][4][(n_rotate-1)-j][i]; //5存4右轉一次的
		for(int i=0;i<n_rotate;i++)
			for(int j=0;j<n_rotate;j++)
				class_matrix_rotate[rotate][6][i][j]=class_matrix_rotate[rotate][5][(n_rotate-1)-j][i]; //6存4右轉兩次的
		for(int i=0;i<n_rotate;i++)
			for(int j=0;j<n_rotate;j++)
				class_matrix_rotate[rotate][7][i][j]=class_matrix_rotate[rotate][6][(n_rotate-1)-j][i]; //7存4右轉三次的
	}

	//隨機擺放選轉後的CM
	//=============================================
	srand(65487);

	//0729新增
	//取得CM處理順序的位置
	//=============================================
	vector<vector<vector<int> > > Pro_Po2(8, vector<vector<int> >(2*2/*CM大小*/, vector<int>(2))); //2
	vector<vector<vector<int> > > Pro_Po4(8, vector<vector<int> >(4*4/*CM大小*/, vector<int>(2))); //4
	vector<vector<vector<int> > > Pro_Po8(8, vector<vector<int> >(8*8/*CM大小*/, vector<int>(2))); //8
	vector<vector<vector<int> > > Pro_Po16(8, vector<vector<int> >(16*16/*CM大小*/, vector<int>(2))); //16

	for(int size=0; size<4 ;size++){ //0=>2 1=>4 2=>8 3=>16
		for(int rotate=0; rotate<8; rotate++){
			if(size==0){
				for(int i=0;i<2;i++){
					for(int j=0;j<2;j++){
						Pro_Po2[rotate][class_matrix_rotate[size][rotate][i][j]][0]=i;
						Pro_Po2[rotate][class_matrix_rotate[size][rotate][i][j]][1]=j;
					}
				}
			}
			else if(size==1){
				for(int i=0;i<4;i++){
					for(int j=0;j<4;j++){
						Pro_Po4[rotate][class_matrix_rotate[size][rotate][i][j]][0]=i;
						Pro_Po4[rotate][class_matrix_rotate[size][rotate][i][j]][1]=j;
					}
				}
			}
			else if(size==2){
				for(int i=0;i<8;i++){
					for(int j=0;j<8;j++){
						Pro_Po8[rotate][class_matrix_rotate[size][rotate][i][j]][0]=i;
						Pro_Po8[rotate][class_matrix_rotate[size][rotate][i][j]][1]=j;
					}
				}
			}else if(size==3){
				for(int i=0;i<16;i++){
					for(int j=0;j<16;j++){
						Pro_Po16[rotate][class_matrix_rotate[size][rotate][i][j]][0]=i;
						Pro_Po16[rotate][class_matrix_rotate[size][rotate][i][j]][1]=j;
					}
				}
			}
		}
	}

	//依照輸入值ImageQuality查表 找出各區塊的標準差閥值
	//=============================================
	int block_thres[4]={0}; // [0]存2, [1]存4, [2]存8, [3]存16 的標準差門檻
	double thres_psnr2 ,thres_psnr4,thres_psnr8,thres_psnr16;
	if(ImageQuality>=85){       //條件上限 (高品質=>全切成2*2大小的區塊)
		block_thres[3]=0;
		block_thres[2]=0;
		block_thres[1]=0;
		block_thres[0]=128;
	}
	else if(ImageQuality<=35/*43*/){  //條件下限 (低品質=>全切成16*16大小的區塊) 
		block_thres[3]=128;
		block_thres[2]=0;
		block_thres[1]=0;
		block_thres[0]=0;
	}
	else{
		for(double i=1;i<128;i++){
			thres_psnr2 = -8.026*log(i) + 92.945;
			thres_psnr4 = -7.363*log(i) + 84.688;
			thres_psnr8 = -7.146*log(i) + 77.924;
			thres_psnr16 = -6.788*log(i) + 70.4;
			if(thres_psnr2>=ImageQuality)
				block_thres[0]=i;
			if(thres_psnr4>=ImageQuality)
				block_thres[1]=i;
			if(thres_psnr8>=ImageQuality)
				block_thres[2]=i;
			if(thres_psnr16>=ImageQuality)
				block_thres[3]=i;
		}
	}

	//預先排列好CM
	int y_s,x_s,y_e,x_e;
	double bs; 
	for(int BS=0; BS<4; BS++){
		if(BS==0)
			bs=2;
		else if(BS==1)
			bs=4;
		else if(BS==2)
			bs=8;
		else if(BS==3)
			bs=16;
		for(double Y=0 ; Y<(m/bs) ; Y++){
			for(double X=0 ; X<(n/bs) ; X++){
				y_s=Y*bs;
				x_s=X*bs;
				y_e=(Y+1)*bs;
				if(y_e>m)
					y_e=m;
				x_e=(X+1)*bs;
				if(x_e>n)
					x_e=n;
				if(BS==0)
					CM_reg2[Y][X]=arrange_CM(m,n,Y,X,y_s,y_e,x_s,x_e,(int)bs,class_matrix,class_matrix_rotate);
				else if(BS==1)
					CM_reg4[Y][X]=arrange_CM(m,n,Y,X,y_s,y_e,x_s,x_e,(int)bs,class_matrix,class_matrix_rotate);
				else if(BS==2)
					CM_reg8[Y][X]=arrange_CM(m,n,Y,X,y_s,y_e,x_s,x_e,(int)bs,class_matrix,class_matrix_rotate);
				else if(BS==3)
					CM_reg16[Y][X]=arrange_CM(m,n,Y,X,y_s,y_e,x_s,x_e,(int)bs,class_matrix,class_matrix_rotate);
			}
		}
	}

	mid1_time = clock(); //(顯示結果用"可刪除")

	//建立積分圖
	/*vector<vector<double>> Integration1(src.rows,vector<double>(src.cols,-1)); //1階動差
	vector<vector<double>> Integration2(src.rows,vector<double>(src.cols,-1)); //2階動差
	for(int i=0; i<src.rows; i++){
		for(int j=0; j<src.cols; j++){
			double Integration1_l=0,Integration1_u=0,Integration1_lu=0;
			double Integration2_l=0,Integration2_u=0,Integration2_lu=0;
			if(i==0&&j==0){
				Integration1[i][j]=src.data[i*src.cols+j];
				Integration2[i][j]=pow(src.data[i*src.cols+j],2.0);
			}
			else{
				if(i-1>=0){
					Integration1_u=Integration1[i-1][j];
					Integration2_u=Integration2[i-1][j];
				}
				if(j-1>=0){
					Integration1_l=Integration1[i][j-1];
					Integration2_l=Integration2[i][j-1];
				}
				if(i-1>=0&&j-1>=0){
					Integration1_lu=Integration1[i-1][j-1];
					Integration2_lu=Integration2[i-1][j-1];
				}
				Integration1[i][j]=src.data[i*src.cols+j]+Integration1_u+Integration1_l-Integration1_lu;
				Integration2[i][j]=pow(src.data[i*src.cols+j],2.0)+Integration2_u+Integration2_l-Integration2_lu;
			}		
		}
	}*/

	//mid1_time = clock();

	//計算區塊大小以及其ab值(可同步)
	//=============================================
	int y_start,x_start,y_end,x_end;                    //設定處裡區塊的範圍,防止超出邊界
	int local_a,local_b;                                //計算不同區塊大小的量化等級
	int count_CM[4]={0};                                //計算使用多少不同大小的區塊(顯示結果用"可刪除")
	double Beta;                                        //量化等級所需補償的Beta值
	double local_mean[4]={0},local_sd[4]={0};           //計算不同區塊大小的平均值,均差平方與標準差
	                                                    //0存區塊大小為2*2的 1存4*4 以此類推
	for(double Y=0 ; Y<(m/16) ; Y++){
		for(double X=0 ; X<(n/16) ; X++){
			double count_difusse,error;
			y_start=Y*16;
			x_start=X*16;
			y_end=(Y+1)*16;
			if(y_end>m)
				y_end=m;
			x_end=(X+1)*16;
			if(x_end>n)
				x_end=n;
			cal_Parameter(y_start,y_end,x_start,x_end,16,src,&local_a,&local_b,&local_mean[3],&local_sd[3]);
			/*test(y_start,y_end,x_start,x_end,16,src,&local_a,&local_b);
			double Integration1_a=0,Integration1_b=0,Integration1_c=Integration1[y_end-1][x_end-1],Integration1_d=0;
			double Integration2_a=0,Integration2_b=0,Integration2_c=Integration2[y_end-1][x_end-1],Integration2_d=0;
			if(y_start-1>=0 && x_start-1>=0){
				Integration1_a=Integration1[y_start-1][x_start-1];
				Integration2_a=Integration2[y_start-1][x_start-1];
			}
			if(y_start-1>=0){
				Integration1_b=Integration1[y_start-1][x_end-1];
				Integration2_b=Integration2[y_start-1][x_end-1];
			}
			if(x_start-1>=0){
				Integration1_d=Integration1[y_end-1][x_start-1];
				Integration2_d=Integration2[y_end-1][x_start-1];
			}
			local_mean[3]=(Integration1_a-Integration1_b+Integration1_c-Integration1_d)/16/16;
			double reg=(Integration2_a-Integration2_b+Integration2_c-Integration2_d)/16/16;
			local_sd[3]=sqrt(reg-pow(local_mean[3],2));*/

			if(local_sd[3]>block_thres[3]){  //區塊標準差大於使用者設定值(alpha)對照的標準差(gamma)
				for(int q=0 ; q<2 ; q++){
					for(int w=0 ; w<2 ; w++){
						y_start=q*8+Y*16;
						x_start=w*8+X*16;
						y_end=(q+1)*8+Y*16;
						if(y_end>m)
							y_end=m;
						x_end=(w+1)*8+X*16;
						if(x_end>n)
							x_end=n;
						cal_Parameter(y_start,y_end,x_start,x_end,8,src,&local_a,&local_b,&local_mean[2],&local_sd[2]);
						/*test(y_start,y_end,x_start,x_end,8,src,&local_a,&local_b);
						Integration1_a=0,Integration1_b=0,Integration1_c=Integration1[y_end-1][x_end-1],Integration1_d=0;
						Integration2_a=0,Integration2_b=0,Integration2_c=Integration2[y_end-1][x_end-1],Integration2_d=0;
						if(y_start-1>=0 && x_start-1>=0){
							Integration1_a=Integration1[y_start-1][x_start-1];
							Integration2_a=Integration2[y_start-1][x_start-1];
						}
						if(y_start-1>=0){
							Integration1_b=Integration1[y_start-1][x_end-1];
							Integration2_b=Integration2[y_start-1][x_end-1];
						}
						if(x_start-1>=0){
							Integration1_d=Integration1[y_end-1][x_start-1];
							Integration2_d=Integration2[y_end-1][x_start-1];
						}
						local_mean[2]=(Integration1_a-Integration1_b+Integration1_c-Integration1_d)/8/8;
						reg=(Integration2_a-Integration2_b+Integration2_c-Integration2_d)/8/8;
						local_sd[2]=sqrt(reg-pow(local_mean[2],2));*/

						if(local_sd[2]>block_thres[2]){
							for(int e=0 ; e<2 ; e++){
								for(int f=0 ; f<2 ; f++){
									y_start=e*4+q*8+Y*16;
									x_start=f*4+w*8+X*16;
									y_end=(e+1)*4+q*8+Y*16;
									if(y_end>m)
										y_end=m;
									x_end=(f+1)*4+w*8+X*16;
									if(x_end>n)
										x_end=n;
									cal_Parameter(y_start,y_end,x_start,x_end,4,src,&local_a,&local_b,&local_mean[1],&local_sd[1]);
									/*test(y_start,y_end,x_start,x_end,4,src,&local_a,&local_b);
									Integration1_a=0,Integration1_b=0,Integration1_c=Integration1[y_end-1][x_end-1],Integration1_d=0;
									Integration2_a=0,Integration2_b=0,Integration2_c=Integration2[y_end-1][x_end-1],Integration2_d=0;
									if(y_start-1>=0 && x_start-1>=0){
										Integration1_a=Integration1[y_start-1][x_start-1];
										Integration2_a=Integration2[y_start-1][x_start-1];
									}
									if(y_start-1>=0){
										Integration1_b=Integration1[y_start-1][x_end-1];
										Integration2_b=Integration2[y_start-1][x_end-1];
									}
									if(x_start-1>=0){
										Integration1_d=Integration1[y_end-1][x_start-1];
										Integration2_d=Integration2[y_end-1][x_start-1];
									}
									local_mean[1]=(Integration1_a-Integration1_b+Integration1_c-Integration1_d)/4/4;
									reg=(Integration2_a-Integration2_b+Integration2_c-Integration2_d)/4/4;
									local_sd[1]=sqrt(reg-pow(local_mean[1],2));*/
									
									if(local_sd[1]>block_thres[1]){
										for(int g=0 ; g<2 ; g++){
											for(int h=0 ; h<2 ; h++){
												y_start=g*2+e*4+q*8+Y*16;
												x_start=h*2+f*4+w*8+X*16;
												y_end=(g+1)*2+e*4+q*8+Y*16;
												if(y_end>m)
													y_end=m;
												x_end=(h+1)*2+f*4+w*8+X*16;
												if(x_end>n)
													x_end=n;
												cal_Parameter(y_start,y_end,x_start,x_end,2,src,&local_a,&local_b,&local_mean[0],&local_sd[0]);
												/*test(y_start,y_end,x_start,x_end,2,src,&local_a,&local_b);
												Integration1_a=0,Integration1_b=0,Integration1_c=Integration1[y_end-1][x_end-1],Integration1_d=0;
												Integration2_a=0,Integration2_b=0,Integration2_c=Integration2[y_end-1][x_end-1],Integration2_d=0;
												if(y_start-1>=0 && x_start-1>=0){
													Integration1_a=Integration1[y_start-1][x_start-1];
													Integration2_a=Integration2[y_start-1][x_start-1];
												}
												if(y_start-1>=0){
													Integration1_b=Integration1[y_start-1][x_end-1];
													Integration2_b=Integration2[y_start-1][x_end-1];
												}
												if(x_start-1>=0){
													Integration1_d=Integration1[y_end-1][x_start-1];
													Integration2_d=Integration2[y_end-1][x_start-1];
												}
												local_mean[0]=(Integration1_a-Integration1_b+Integration1_c-Integration1_d)/2/2;
												reg=(Integration2_a-Integration2_b+Integration2_c-Integration2_d)/2/2;
												local_sd[0]=sqrt(reg-pow(local_mean[0],2));*/

												Beta=beta2[(int)local_sd[0]];
												local_a=local_a+(local_mean[0]-local_a)*Beta;
												local_b=local_b-(local_b-local_mean[0])*Beta;
												for(int memberIndex=0; memberIndex<2*2; memberIndex++){
													int	ni=y_start+Pro_Po2[CM_reg2[y_start/2][x_start/2]][memberIndex][0];
													int nj=x_start+Pro_Po2[CM_reg2[y_start/2][x_start/2]][memberIndex][1];
													if(ni>=0&&ni<src.rows&&nj>=0&&nj<src.cols){
														count_difusse=0;
														for(int i=-1 ; i<=1 ; i++)
															for(int j=-1 ; j<=1 ; j++)
																if( (ni+i)>=0 && (nj+j)>=0 && (ni+i)<m && (nj+j)<n && class_matrix[0][ni+i][nj+j]>memberIndex )
																	count_difusse=count_difusse+diffused_matrix2[i+1][j+1];
														if(image_reg[ni][nj]<local_mean[0])
															image_out[ni][nj]=local_a;
														else
															image_out[ni][nj]=local_b;
														error=image_reg[ni][nj]-image_out[ni][nj];
														for(int i=-1 ; i<=1 ; i++)
															for(int j=-1 ; j<=1 ; j++)
																if( (ni+i)>=0 && (nj+j)>=0 && (ni+i)<m && (nj+j)<n && count_difusse!=0 )
																	image_reg[ni+i][nj+j]=image_reg[ni+i][nj+j]+(error*diffused_matrix2[i+1][j+1])/count_difusse;
													}
												}
												count_CM[0]++; //(顯示結果用"可刪除")
											}
										}
									}
									else{
										Beta=beta4[(int)local_sd[1]];
										local_a=local_a+(local_mean[1]-local_a)*Beta;
										local_b=local_b-(local_b-local_mean[1])*Beta;
										for(int memberIndex=0; memberIndex<4*4; memberIndex++){
											int	ni=y_start+Pro_Po4[CM_reg4[y_start/4][x_start/4]][memberIndex][0];
											int nj=x_start+Pro_Po4[CM_reg4[y_start/4][x_start/4]][memberIndex][1];
											if(ni>=0&&ni<src.rows&&nj>=0&&nj<src.cols){
													count_difusse=0;
												for(int i=-1 ; i<=1 ; i++)
													for(int j=-1 ; j<=1 ; j++)
														if( (ni+i)>=0 && (nj+j)>=0 && (ni+i)<m && (nj+j)<n && class_matrix[1][ni+i][nj+j]>memberIndex )
															count_difusse=count_difusse+diffused_matrix4[i+1][j+1];
												if(image_reg[ni][nj]<local_mean[1])
													image_out[ni][nj]=local_a;
												else
													image_out[ni][nj]=local_b;
												error=image_reg[ni][nj]-image_out[ni][nj];
												for(int i=-1 ; i<=1 ; i++)
													for(int j=-1 ; j<=1 ; j++)
														if( (ni+i)>=0 && (nj+j)>=0 && (ni+i)<m && (nj+j)<n && count_difusse!=0 )
															image_reg[ni+i][nj+j]=image_reg[ni+i][nj+j]+(error*diffused_matrix4[i+1][j+1])/count_difusse;
											}
										}
										count_CM[1]++; //(顯示結果用"可刪除")
									}
								}
							}
						}
						else{
							Beta=beta8[(int)local_sd[2]];
							local_a=local_a+(local_mean[2]-local_a)*Beta;
							local_b=local_b-(local_b-local_mean[2])*Beta;
							for(int memberIndex=0; memberIndex<8*8; memberIndex++){
								int	ni=y_start+Pro_Po8[CM_reg8[y_start/8][x_start/8]][memberIndex][0];
								int nj=x_start+Pro_Po8[CM_reg8[y_start/8][x_start/8]][memberIndex][1];
								if(ni>=0&&ni<src.rows&&nj>=0&&nj<src.cols){
									count_difusse=0;
									for(int i=-1 ; i<=1 ; i++)
										for(int j=-1 ; j<=1 ; j++)
											if( (ni+i)>=0 && (nj+j)>=0 && (ni+i)<m && (nj+j)<n && class_matrix[2][ni+i][nj+j]>memberIndex )
												count_difusse=count_difusse+diffused_matrix8[i+1][j+1];
									if(image_reg[ni][nj]<local_mean[2])
										image_out[ni][nj]=local_a;
									else
										image_out[ni][nj]=local_b;
									error=image_reg[ni][nj]-image_out[ni][nj];
									for(int i=-1 ; i<=1 ; i++)
										for(int j=-1 ; j<=1 ; j++)
											if( (ni+i)>=0 && (nj+j)>=0 && (ni+i)<m && (nj+j)<n && count_difusse!=0 )
												image_reg[ni+i][nj+j]=image_reg[ni+i][nj+j]+(error*diffused_matrix8[i+1][j+1])/count_difusse;
								}
							}
							count_CM[2]++; //(顯示結果用"可刪除")
						}
					}
				}
			}
			else{
				Beta=beta16[(int)local_sd[3]];
				local_a=local_a+(local_mean[3]-local_a)*Beta;
				local_b=local_b-(local_b-local_mean[3])*Beta;
				for(int memberIndex=0; memberIndex<16*16; memberIndex++){
					int	ni=y_start+Pro_Po16[CM_reg16[y_start/16][x_start/16]][memberIndex][0];
					int nj=x_start+Pro_Po16[CM_reg16[y_start/16][x_start/16]][memberIndex][1];
					if(ni>=0&&ni<src.rows&&nj>=0&&nj<src.cols){
						count_difusse=0;
						for(int i=-1 ; i<=1 ; i++)
							for(int j=-1 ; j<=1 ; j++)
								if( (ni+i)>=0 && (nj+j)>=0 && (ni+i)<m && (nj+j)<n && class_matrix[3][ni+i][nj+j]>memberIndex )
									count_difusse=count_difusse+diffused_matrix16[i+1][j+1];
						if(image_reg[ni][nj]<local_mean[3])
							image_out[ni][nj]=local_a;
						else
							image_out[ni][nj]=local_b;
						error=image_reg[ni][nj]-image_out[ni][nj];
						for(int i=-1 ; i<=1 ; i++)
							for(int j=-1 ; j<=1 ; j++)
								if( (ni+i)>=0 && (nj+j)>=0 && (ni+i)<m && (nj+j)<n && count_difusse!=0 )
									image_reg[ni+i][nj+j]=image_reg[ni+i][nj+j]+(error*diffused_matrix16[i+1][j+1])/count_difusse;
					}
				}
				count_CM[3]++; //(顯示結果用"可刪除")
			}
		}
	}

	//}
	mid_end_time = clock(); //(顯示結果用"可刪除")
	

	//影像輸出
	//=============================================
	cv::Mat tdst(src.rows,src.cols,CV_8UC1);
	for(int i=0 ; i<src.rows ; i++ )
		for(int j=0 ; j<src.cols ; j++)
			tdst.data[i*src.cols+j]=image_out[i][j];
	dst=tdst.clone();

	//計算有無限core時最佳速度 (顯示結果用"可刪除")
	final_end_time = clock();
	//total_time = (double)(1000000*(mid_end_time - mid1_time)/(m*n/16/16))/CLOCKS_PER_SEC; // 無限核
	total_time = (double)(mid_end_time - mid1_time)/CLOCKS_PER_SEC;
	printf("Total time =%f \n",total_time);
	
}
