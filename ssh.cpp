/************************* DISCLAIMER **********************************/
/*
This SSH code is copyright protected © 2017 by Chen Luo and Anshumali Shrivastava
For the details about SSH, please refer the paper:
Chen Luo and Anshumail Shrivastava, "SSH (Sketch, Shingle, & Hash) for Indexing Massive-Scale Time Series" NIPS Time Series Workshop.
*/
/***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <string.h>
#include <list>

#define INF 1e20       //Pseudo Infitinte number for this code
#define dist(x,y) ((x-y)*(x-y))

using namespace std;

typedef struct Set
{
   int* weights; // list of the number of the patterns happened.
   int**  patterns; // list of the patterns.
   int length; // the length of the Set

} Set;

typedef struct HashPair
{
    int k; // index of the sample
    int y; // value that between [0,x_k]
} HashPair;


typedef struct HashTable
{
    std::list<int> ** HashBuckets; // Hash Buckets in the hashtables
    int length; // number of buckets in the hash table
} HashTable;


/// If serious error happens, terminate the program.
void error(int id)
{
    if(id==1)
        printf("ERROR : Memory can't be allocated!!!\n\n");
    else if ( id == 2 )
        printf("ERROR : File not Found!!!\n\n");
    else if ( id == 3 )
        printf("ERROR : Can't create Output File!!!\n\n");
    else if ( id == 4 )
    {
        printf("ERROR: Invalid Number of Arguments!!!\n");
        printf("Command Usage:   UCR_ED.exe  data_file  query_file   m   \n");
        printf("For example  :   UCR_ED.exe  data.txt   query.txt   128  \n");
    }
    exit(1);
}

// Generate a random number based on the mean and variance
double randn (double mu, double sigma)
{
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;
 
  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }
 
  do
    {
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (double) X1);
}

// generate and iid random filter r
void iid (int length, double * r)
{
    for (int i=0; i<length; ++i) {
        double number = randn(0,1);
        r[i] = number ;
    }
}

// calculate the sign of the inner product between r and d
double sign (double *r, double * d, int l)
{
    double sum = 0.0;
    for (int i=0;i<l;i++)
    {
        sum += r[i]*d[i];
    }
    if (sum>=0) return 1;
    else return -1;

}

// Do n-gram on Db
void ngram (int * Db, int k, int ** set, int m)
{

    for (int i=0;i<m;i++)
    {
        int idx = 0;
        for (int j=i;j < i+k;j++)
        {
            set[i][idx++] = Db[j];
        }
    }

}

// Evaluate whether two sets are equal or not
bool setequal (int * d1, int * d2, int k) // check whether two objects are equal
{
    for (int i=0;i<k;i++)
    {
        if (d1[i]!=d2[i]) return false;
    }
    
    return true;
}

// Check whether a pattern is in a set
int inSet (Set * set,int * d, int k, int s) // Check wether d in set.
{
    for (int i=0;i< s;i++)
    {
        if (setequal(set->patterns[i],d,k)) 
        {
            return i;
        }
    }
    return -1;
}

//Check whether an item d is in the set list
int inList (int * set, int d, int s)
{
    for (int i=0;i<s;i++)
    {
        if (set[i] == d) return i;
    }
    return -1;
}

// Create the weighted set
Set * set_create(int **Ds, int k, int s) // k is the object length, s is the set size
{
    Set *set =  (Set*)malloc(sizeof(Set)) ;
    // Initialize the set
    set->patterns = (int **) malloc(s*sizeof(int*));
    for (int t=0;t<s;t++)
    {
        set->patterns[t] = (int*)malloc((k)*sizeof(int)) ;
    }
    set->weights = (int *) malloc(s*sizeof(int)) ;
    set->length = 0;

    int index = 0;

    for (int i=0;i<s;i++)
    {
        index = inSet(set, Ds[i], k, s);
        if ( index!= -1)
        {
            set->weights[index] ++ ;
        }
        else
        {
            for (int j=0;j<k;j++)
            {
               set->patterns[set->length][j] = Ds[i][j] ;
            }

            set->weights[set->length] = 1;
            set->length ++ ;
        }
    }
    return set;

}

// Print set
void print_set(Set *set)
{
    for (int i=0;i<set->length;i++)
    {
        printf("%d  ", set->weights[i]);
    }
    printf("\n");
}

// Calculate the Jaccard Similarity between two set1 and set2
double jsim (int ** set1, int ** set2, int k, int s) // k is the object length, s is the set size
{

    Set * s1 = set_create(set1, k, s) ;
    Set * s2 = set_create(set2, k, s) ;

    double * mins = (double*)malloc((s1->length + s2->length)*sizeof(double)) ;
    double * maxs = (double*)malloc((s1->length + s2->length)*sizeof(double)) ;

    Set* set = (Set*)malloc(sizeof(Set)) ;
    set->patterns = (int **) malloc((s1->length + s2->length)*sizeof(int*));

    for (int t=0;t<(s1->length + s2->length);t++)
    {
        set->patterns[t] = (int*)malloc((k)*sizeof(int)) ;
    }
    set->weights = (int *) malloc((s1->length + s2->length)*sizeof(int)) ;
    set->length = 0;

    // Add s2 into set list
    for (int i=0;i<s2->length;i++)
    {
        for (int j=0;j<k;j++)
        {
           set->patterns[i][j] = s2->patterns[i][j] ;
        }
        // Give them min value.
        maxs[i] = s2->weights[i] ;
        set->length ++ ;
    }

    // Add s1 into set list
    int index = 0;
    for (int i=0;i<s1->length;i++)
    {
        index = inSet(set, s1->patterns[i], k, (s1->length + s2->length));
        if ( index!= -1)
        {
            if (s1->weights[index] < maxs[index]) mins[index]=s1->weights[index] ;
            else 
            {
                mins[index] = maxs[index] ;
                maxs[index] = s1->weights[index];
            }
        }
        else
        {
            for (int j=0;j<k;j++)
            {
               set->patterns[set->length][j] = s1->patterns[i][j] ;
            }

            maxs[set->length] = s1->weights[i];
            set->length ++ ;
        }
    }

    double max = 0.0 ;
    double min = 0.0 ;

    for (int i=0;i<set->length;i++)
    {
        max += maxs[i] ;
        min += mins[i] ;
    }

    return min/max;
}

// generate a random number from a uniform distribution
double uniform(double a, double b)
{  
    return rand() / (RAND_MAX + 1.0) * (b - a) + a;
}

// Generate a random number from a gaussion distribution
double gauss(double mu,double sigma)
{
    double x1, x2, w, y1, y2;

    do {
        x1 = 2.0 * uniform(0,1) - 1.0;
        x2 = 2.0 * uniform(0,1) - 1.0;
        w = x1 * x1 + x2 * x2;
    } while ( w >= 1.0 );

    w = sqrt( (-2.0 * log( w ) ) / w );
    y1 = x1 * w;
    y2 = x2 * w;
    return mu+sigma*y1; 
}

// Generate a random number from a gamma distribution
double rgamma(double a,double b) 
{
    double d,c,x,v,u;
    if(a>=1)
    {
        d = a-1./3.; 
        c = 1./sqrt(9.*d);
        while(1)
        {
            do 
            {
                x=gauss(0,1.0); 
                v=1.+c*x;
            } 
            while(v<=0.);

            v = v * v * v; 
            u = uniform(0,1);
            if( u < 1.0-0.0331*(x*x)*(x*x) )
            {
                return d*v*b;
            }
            if( log(u) < 0.5*x*x+d*(1.0-v+log(v)) )
            {
                return d*v*b;
            }
        }
    } else 
    {
        x = rgamma(a+1,b);
        x = x * pow(uniform(0,1), 1.0/a); 
        return x;
    }
}

// Calculate argmin
int argmin (double * a, int n)
{
    int k = 0;
    double min = a[0];
    for (int i=0;i<n;i++)
    {
        if (min > a[i]) k = i;
    }
    return k;
}

/*
The function CWS is Consistent weighted sampling, for the details of this function, please refer the following paper:
Ioffe, Sergey. "Improved consistent sampling, weighted minhash and l1 sketching." Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 2010.
*/
int CWS (Set * set)
{

    double *r = (double*)malloc(set->length*sizeof(double)) ; 
    double *c = (double*)malloc(set->length*sizeof(double)) ; 
    double *b = (double*)malloc(set->length*sizeof(double)) ; 

    double *t = (double*)malloc(set->length*sizeof(double)) ; 
    double *y = (double*)malloc(set->length*sizeof(double)) ; 
    double *a = (double*)malloc(set->length*sizeof(double)) ; 

    HashPair * pair = (HashPair*)malloc(sizeof(HashPair)) ; 

    
    //print_set(set);

    int i;
    for (int i=0;i<set->length;i++)
    {
        r[i] = rgamma(2.0,1.0);
        c[i] = rgamma(2.0,1.0);
        b[i] = uniform(0.0,1);
    }
    i = 0;
    while(set->weights[i] > 0 && i<set->length)
    {
        double s = (double) set->weights[i];
        t[i] = (log(s)/r[i]) + b[i] ;

        y[i] = exp(r[i]*(t[i]-b[i]));
        a[i] = c[i]/(y[i]*exp(r[i])) ;

        i++;
    }

    pair->k = argmin(a, set->length);
    pair->y = (int) y[pair->k];

    return pair->k;
}

/// Calculate Dynamic Time Wrapping distance
/// A,B: data and query, respectively
/// cb : cummulative bound used for early abandoning
/// r  : size of Sakoe-Chiba warpping band
double dtw(double* A, double* B, double *cb, int m, int r, double bsf = INF)
{

    double *cost;
    double *cost_prev;
    double *cost_tmp;
    int i,j,k;
    double x,y,z,min_cost;

    /// Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(r).
    cost = (double*)malloc(sizeof(double)*(2*r+1));
    for(k=0; k<2*r+1; k++)    cost[k]=INF;

    cost_prev = (double*)malloc(sizeof(double)*(2*r+1));
    for(k=0; k<2*r+1; k++)    cost_prev[k]=INF;

    for (i=0; i<m; i++)
    {
        k = max(0,r-i);
        min_cost = INF;

        for(j=max(0,i-r); j<=min(m-1,i+r); j++, k++)
        {
            /// Initialize all row and column
            if ((i==0)&&(j==0))
            {
                cost[k]=dist(A[0],B[0]);
                min_cost = cost[k];
                continue;
            }

            if ((j-1<0)||(k-1<0))     y = INF;
            else                      y = cost[k-1];
            if ((i-1<0)||(k+1>2*r))   x = INF;
            else                      x = cost_prev[k+1];
            if ((i-1<0)||(j-1<0))     z = INF;
            else                      z = cost_prev[k];

            /// Classic DTW calculation
            cost[k] = min( min( x, y) , z) + dist(A[i],B[j]);

            /// Find minimum cost in row for early abandoning (possibly to use column instead of row).
            if (cost[k] < min_cost)
            {   min_cost = cost[k];
            }
        }

        /// We can abandon early if the current cummulative distace with lower bound together are larger than bsf
        if (i+r < m-1 && min_cost + cb[i+r+1] >= bsf)
        {   free(cost);
            free(cost_prev);
            return min_cost + cb[i+r+1];
        }

        /// Move current array to previous array.
        cost_tmp = cost;
        cost = cost_prev;
        cost_prev = cost_tmp;
    }
    k--;

    /// the DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array.
    double final_dtw = cost_prev[k];
    free(cost);
    free(cost_prev);
    return final_dtw;
}

int main(int argc , char *argv[])
{
    FILE *fp_data;              // the input file pointer
    FILE *fp_query;
    FILE *fp_result;
    int m;                 // the length of the time series. 1639 in current data set
    int n;                 // the number of time series
    int w;
    int delta;
    int k;
    int scband = 10;
    int hash_num = 20;

    char d[20000];
    
    if (argc<=2)      
        error(4);

    fp_data = fopen(argv[1],"r");
    if( fp_data == NULL )
        exit(2);

    fp_query = fopen(argv[2],"r");
    if( fp_query == NULL )
        exit(2);

    m = atol(argv[3]);
    n = atol(argv[4]);
    w = atol(argv[5]);
    delta = atol(argv[6]);
    k = atol(argv[7]);
    scband = atol(argv[8]);

    int i = 0 ;
    int j = 0 ;


    // Extract the time series data set
    double ** D = (double**)malloc(n*sizeof(double*)) ; // Initialize Datas
    for (int k=0;k<n;k++)
    {
        D[k] = (double*)malloc(m*sizeof(double)) ;
    }

    while(fscanf(fp_data,"%[^\n]\n",d) != EOF && j< n)
    {
        
        char *p =  strtok(d,",");
        i = 0;
        p = strtok (NULL, ",");
        p = strtok (NULL, ",");
        while (p != NULL && i<m)
        {
            D[j][i++] = atof(p);
            p = strtok (NULL, ",");
        }
        j++;
    }
    

    printf("Datasets Loaded!\n");

    //Start to do window slide
    //int w = 100;
    //int delta = 3; 
    double *r = (double*)malloc(w*sizeof(double)) ; // initialize the random vector r iid
    iid(w,r) ;

    int ** Db = (int**)malloc(n*sizeof(int*)) ; // Initialize Data

    for (int k=0;k<n;k++)
    {
        Db[k] = (int*)malloc(((m-w)/delta)*sizeof(int)) ;
    }

    double *subs = (double*)malloc(w*sizeof(double)) ;
    for (i=0;i<n;i++)
    {
        int idxb = 0;
        for (j = 0;j<m-w;j+=delta)
        {
            int idx = 0;
            for (int t=j;t<j+w;t++)
            {
                subs[idx++] = D[i][t] ;
            }

            int bit = sign(r,subs,w) ;
            
            Db[i][idxb++] = bit ;
        }
    }
    printf("Sliding Windows bit Extraction Finished!\n");
    
    // Do n-gram here
    //int k = 15; // length of doing n-gram
    int length = (m-w)/delta - k;

    Set **Ds = (Set**)malloc(sizeof(Set*)) ;

    for (i=0;i<n;i++)
    {
        int ** Ds_org = (int**)malloc((length)*sizeof(int*)) ; // Initialize Datas
        for (int t=0;t<(length);t++)
        {
            Ds_org [t] = (int*)malloc((k)*sizeof(int)) ;
        }   
        ngram(Db[i], k, Ds_org, length);
        Ds[i] = set_create(Ds_org, k, length) ;   
    }

    printf("N-gram Generation Finished!\n");

    // Do weighted Minwise Hashing

    //int hash_num = 20;
    int bucket_num = length; // Use the number of n-gram set as the number of buckets

    HashTable * tables = (HashTable *)malloc((hash_num)*sizeof(HashTable));
    for (i = 0;i<hash_num;i++)
    {
        tables[i].length = bucket_num;
        tables[i].HashBuckets = (list<int> **)malloc(bucket_num*sizeof(list<int>*));
        for (j=0;j<bucket_num;j++)
        {
            tables[i].HashBuckets[j] = new list<int>[n]();
        }
    }

    // Insert indexes into Hash tables
    for (i=2;i<n;i++)
    {
        for (int j=0;j<hash_num;j++)
        {         
            int index = CWS(Ds[i]) ;
            tables[j].HashBuckets[index]->push_back(i);
        }
    }

    printf("Weighted Minwise Hashing Finished, Hashed into %d Hashtables!\n", hash_num);

    // Here we do time series query.

    printf("Start Query ......\n");
    // Extract the query time series
    double * Q = (double*)malloc(m*sizeof(double)) ; // Initialize Datas
    j = 0;
    while(fscanf(fp_query,"%[^\n]\n",d) != EOF && j< n)
    {

        char *p =  strtok(d,",");
        i = 0;
        p = strtok (NULL, ",");
        while (p != NULL && i<m)
        {
            Q[i++] = atof(p);
            p = strtok (NULL, ",");
        }
        j++;
    }

    clock_t begin_time = clock();
    // Sliding Window Local Bit Profile Extraction
    double *subs_q = (double*)malloc(w*sizeof(double)) ;
    int *Db_q = (int*)malloc(((m-w)/delta)*sizeof(int)) ;

    int idxb = 0;
    for (j = 0;j<m-w;j+=delta)
    {
        int idx = 0;
        for (int t=j;t<j+w;t++)
        {
            subs_q[idx++] = Q[t] ;
        }
        int bit = sign(r,subs_q,w) ;
        Db_q[idxb++] = bit ;
    }
    
    //Do n-gram on query time series
    Set *Ds_q = (Set*)malloc(sizeof(Set*)) ;

    int ** Ds_org = (int**)malloc((length)*sizeof(int*)) ; // Initialize Datas
    for (int t=0;t<(length);t++)
    {
        Ds_org [t] = (int*)malloc((k)*sizeof(int)) ;
    }   
    ngram(Db_q, k, Ds_org, length);
    Ds_q = set_create(Ds_org, k, length) ;

    // Map to different hash tables
    int * search_s = (int *)malloc((n)*sizeof(int));
    int search_n = 0;
    for (int j=0;j<hash_num;j++)
    {
        int index = CWS(Ds_q) ;
        list<int>::iterator it;
        for(it = tables[j].HashBuckets[index]->begin();it!=tables[j].HashBuckets[index]->end();it++)
        {
            if (inList(search_s, *it, search_n) == -1)  
            {
                search_s[search_n++] = (*it);
            }
        }   
    }
    
    // Search time series using Branch and Bound (lb_keogh)
    //int scband = 10;

    double best = INF ;
    int best_index = 0;
    for (int i=0;i<search_n;i++)
    {
        double * U = (double*)malloc((m)*sizeof(double)) ;
        double * L = (double*)malloc((m)*sizeof(double)) ;
        double lb_keogh = 0.0;

        for (int j=0;j<m;j++)
        {
            U[j] = -INF;
            L[j] = INF;
            for (int t = j-scband>0?(j-scband):0; t< j+scband;t++)
            {
                if (U[j] < Q[t]) U[j] = Q[t];
                if (L[j] > Q[t]) L[j] = Q[t];
            }
            double c = D[search_s[i]][j] ;
            if (c > U[j]) lb_keogh += (c-U[j])*(c-U[j]) ;
            else if( c < L[j]) lb_keogh += (c-L[j])*(c-L[j]) ;
        }
        lb_keogh = sqrt(lb_keogh);

        double dist = 0.0;
        if (lb_keogh < best)
        {
            dist = dtw(D[search_s[i]], Q, Q, m, scband);
            if (dist < best) 
            {
                best_index = search_s[i];
                best = dist;
            }
        }
    }
    clock_t end_time = clock();
    double time_spent = (double)(end_time - begin_time) / CLOCKS_PER_SEC;

    printf("Query Finished!\n");
    printf("The nereast neighbor time series is: %d, Query time: %f seconds\n", best_index, time_spent);
    
    fclose(fp_data);
    fclose(fp_query);
    
    return 0;
}
