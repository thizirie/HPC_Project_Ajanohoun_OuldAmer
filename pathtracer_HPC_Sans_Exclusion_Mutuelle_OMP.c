/* basé sur on smallpt, a Path Tracer by Kevin Beason, 2008
 *  	http://www.kevinbeason.com/smallpt/ 
 *
 * Converti en C et modifié par Charles Bouillaguet, 2019
 *
 * Pour des détails sur le processus de rendu, lire :
 * 	https://docs.google.com/open?id=0B8g97JkuSSBwUENiWTJXeGtTOHFmSm51UC01YWtCZw
 */

#define _XOPEN_SOURCE
#include <math.h>   
#include <stdlib.h> 
#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>
#include <sys/stat.h>  /* pour mkdir    */ 
#include <unistd.h>    /* pour getuid   */
#include <sys/types.h> /* pour getpwuid */
#include <pwd.h>       /* pour getpwuid */

#include <mpi.h>
#include <string.h>


enum Refl_t {DIFF, SPEC, REFR};   /* types de matériaux (DIFFuse, SPECular, REFRactive) */

struct Sphere { 
	double radius; 
	double position[3];
	double emission[3];     /* couleur émise (=source de lumière) */
	double color[3];        /* couleur de l'objet RGB (diffusion, refraction, ...) */
	enum Refl_t refl;       /* type de reflection */
	double max_reflexivity;
};

static const int KILL_DEPTH = 7;
static const int SPLIT_DEPTH = 4;

/* la scène est composée uniquement de spheres */
struct Sphere spheres[] = { 
// radius position,                         emission,     color,              material 
   {1e5,  { 1e5+1,  40.8,       81.6},      {},           {.75,  .25,  .25},  DIFF, -1}, // Left 
   {1e5,  {-1e5+99, 40.8,       81.6},      {},           {.25,  .25,  .75},  DIFF, -1}, // Right 
   {1e5,  {50,      40.8,       1e5},       {},           {.75,  .75,  .75},  DIFF, -1}, // Back 
   {1e5,  {50,      40.8,      -1e5 + 170}, {},           {},                 DIFF, -1}, // Front 
   {1e5,  {50,      1e5,        81.6},      {},           {0.75, .75,  .75},  DIFF, -1}, // Bottom 
   {1e5,  {50,     -1e5 + 81.6, 81.6},      {},           {0.75, .75,  .75},  DIFF, -1}, // Top 
   {16.5, {40,      16.5,       47},        {},           {.999, .999, .999}, SPEC, -1}, // Mirror 
   {16.5, {73,      46.5,       88},        {},           {.999, .999, .999}, REFR, -1}, // Glass 
   {10,   {15,      45,         112},       {},           {.999, .999, .999}, DIFF, -1}, // white ball
   {15,   {16,      16,         130},       {},           {.999, .999, 0},    REFR, -1}, // big yellow glass
   {7.5,  {40,      8,          120},        {},           {.999, .999, 0   }, REFR, -1}, // small yellow glass middle
   {8.5,  {60,      9,          110},        {},           {.999, .999, 0   }, REFR, -1}, // small yellow glass right
   {10,   {80,      12,         92},        {},           {0, .999, 0},       DIFF, -1}, // green ball
   {600,  {50,      681.33,     81.6},      {12, 12, 12}, {},                 DIFF, -1},  // Light 
   {5,    {50,      75,         81.6},      {},           {0, .682, .999}, DIFF, -1}, // occlusion, mirror
}; 


/********** micro BLAS LEVEL-1 + quelques fonctions non-standard **************/
static inline void copy(const double *x, double *y)
{
	for (int i = 0; i < 3; i++)
		y[i] = x[i];
} 

static inline void zero(double *x)
{
	for (int i = 0; i < 3; i++)
		x[i] = 0;
} 

static inline void axpy(double alpha, const double *x, double *y)
{
	for (int i = 0; i < 3; i++)
		y[i] += alpha * x[i];
} 

static inline void scal(double alpha, double *x)
{
	for (int i = 0; i < 3; i++)
		x[i] *= alpha;
} 

static inline double dot(const double *a, const double *b)
{ 
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
} 

static inline double nrm2(const double *a)
{
	return sqrt(dot(a, a));
}

/********* fonction non-standard *************/
static inline void mul(const double *x, const double *y, double *z)
{
	for (int i = 0; i < 3; i++)
		z[i] = x[i] * y[i];
} 

static inline void normalize(double *x)
{
	scal(1 / nrm2(x), x);
}

/* produit vectoriel */
static inline void cross(const double *a, const double *b, double *c)
{
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

/****** tronque *************/
static inline void clamp(double *x)
{
	for (int i = 0; i < 3; i++) {
		if (x[i] < 0)
			x[i] = 0;
		if (x[i] > 1)
			x[i] = 1;
	}
} 

/******************************* calcul des intersections rayon / sphere *************************************/
   
// returns distance, 0 if nohit 
double sphere_intersect(const struct Sphere *s, const double *ray_origin, const double *ray_direction)
{ 
	double op[3];
	// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
	copy(s->position, op);
	axpy(-1, ray_origin, op);
	double eps = 1e-4;
	double b = dot(op, ray_direction);
	double discriminant = b * b - dot(op, op) + s->radius * s->radius; 
	if (discriminant < 0)
		return 0;   /* pas d'intersection */
	else 
		discriminant = sqrt(discriminant);
	/* détermine la plus petite solution positive (i.e. point d'intersection le plus proche, mais devant nous) */
	double t = b - discriminant;
	if (t > eps) {
		return t;
	} else {
		t = b + discriminant;
		if (t > eps)
			return t;
		else
			return 0;  /* cas bizarre, racine double, etc. */
	}
}

/* détermine si le rayon intersecte l'une des spere; si oui renvoie true et fixe t, id */
bool intersect(const double *ray_origin, const double *ray_direction, double *t, int *id)
{ 
	int n = sizeof(spheres) / sizeof(struct Sphere);
	double inf = 1e20; 
	*t = inf;
	for (int i = 0; i < n; i++) {
		double d = sphere_intersect(&spheres[i], ray_origin, ray_direction);
		if ((d > 0) && (d < *t)) {
			*t = d;
			*id = i;
		} 
	}
	return *t < inf;
} 

/* calcule (dans out) la lumiance reçue par la camera sur le rayon donné */
void radiance(const double *ray_origin, const double *ray_direction, int depth, unsigned short *PRNG_state, double *out)
{ 
	int id = 0;                             // id de la sphère intersectée par le rayon
	double t;                               // distance à l'intersection
	if (!intersect(ray_origin, ray_direction, &t, &id)) {
		zero(out);    // if miss, return black 
		return; 
	}
	const struct Sphere *obj = &spheres[id];
	
	/* point d'intersection du rayon et de la sphère */
	double x[3];
	copy(ray_origin, x);
	axpy(t, ray_direction, x);
	
	/* vecteur normal à la sphere, au point d'intersection */
	double n[3];  
	copy(x, n);
	axpy(-1, obj->position, n);
	normalize(n);
	
	/* vecteur normal, orienté dans le sens opposé au rayon 
	   (vers l'extérieur si le rayon entre, vers l'intérieur s'il sort) */
	double nl[3];
	copy(n, nl);
	if (dot(n, ray_direction) > 0)
		scal(-1, nl);
	
	/* couleur de la sphere */
	double f[3];
	copy(obj->color, f);
	double p = obj->max_reflexivity;

	/* processus aléatoire : au-delà d'une certaine profondeur,
	   décide aléatoirement d'arrêter la récusion. Plus l'objet est
	   clair, plus le processus a de chance de continuer. */
	depth++;
	if (depth > KILL_DEPTH) {
		if (erand48(PRNG_state) < p) {
			scal(1 / p, f); 
		} else {
			copy(obj->emission, out);
			return;
		}
	}

	/* Cas de la réflection DIFFuse (= non-brillante). 
	   On récupère la luminance en provenance de l'ensemble de l'univers. 
	   Pour cela : (processus de monte-carlo) on choisit une direction
	   aléatoire dans un certain cone, et on récupère la luminance en 
	   provenance de cette direction. */
	if (obj->refl == DIFF) {
		double r1 = 2 * M_PI * erand48(PRNG_state);  /* angle aléatoire */
		double r2 = erand48(PRNG_state);             /* distance au centre aléatoire */
		double r2s = sqrt(r2); 
		
		double w[3];   /* vecteur normal */
		copy(nl, w);
		
		double u[3];   /* u est orthogonal à w */
		double uw[3] = {0, 0, 0};
		if (fabs(w[0]) > .1)
			uw[1] = 1;
		else
			uw[0] = 1;
		cross(uw, w, u);
		normalize(u);
		
		double v[3];   /* v est orthogonal à u et w */
		cross(w, u, v);
		
		double d[3];   /* d est le vecteur incident aléatoire, selon la bonne distribution */
		zero(d);
		axpy(cos(r1) * r2s, u, d);
		axpy(sin(r1) * r2s, v, d);
		axpy(sqrt(1 - r2), w, d);
		normalize(d);
		
		/* calcule récursivement la luminance du rayon incident */
		double rec[3];
		radiance(x, d, depth, PRNG_state, rec);
		
		/* pondère par la couleur de la sphère, prend en compte l'emissivité */
		mul(f, rec, out);
		axpy(1, obj->emission, out);
		return;
	}

	/* dans les deux autres cas (réflection parfaite / refraction), on considère le rayon
	   réfléchi par la spère */

	double reflected_dir[3];
	copy(ray_direction, reflected_dir);
	axpy(-2 * dot(n, ray_direction), n, reflected_dir);

	/* cas de la reflection SPEculaire parfaire (==mirroir) */
	if (obj->refl == SPEC) { 
		double rec[3];
		/* calcule récursivement la luminance du rayon réflechi */
		radiance(x, reflected_dir, depth, PRNG_state, rec);
		/* pondère par la couleur de la sphère, prend en compte l'emissivité */
		mul(f, rec, out);
		axpy(1, obj->emission, out);
		return;
	}

	/* cas des surfaces diélectriques (==verre). Combinaison de réflection et de réfraction. */
	bool into = dot(n, nl) > 0;      /* vient-il de l'extérieur ? */
	double nc = 1;                   /* indice de réfraction de l'air */
	double nt = 1.5;                 /* indice de réfraction du verre */
	double nnt = into ? (nc / nt) : (nt / nc);
	double ddn = dot(ray_direction, nl);
	
	/* si le rayon essaye de sortir de l'objet en verre avec un angle incident trop faible,
	   il rebondit entièrement */
	double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
	if (cos2t < 0) {
		double rec[3];
		/* calcule seulement le rayon réfléchi */
		radiance(x, reflected_dir, depth, PRNG_state, rec);
		mul(f, rec, out);
		axpy(1, obj->emission, out);
		return;
	}
	
	/* calcule la direction du rayon réfracté */
	double tdir[3];
	zero(tdir);
	axpy(nnt, ray_direction, tdir);
	axpy(-(into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)), n, tdir);

	/* calcul de la réflectance (==fraction de la lumière réfléchie) */
	double a = nt - nc;
	double b = nt + nc;
	double R0 = a * a / (b * b);
	double c = 1 - (into ? -ddn : dot(tdir, n));
	double Re = R0 + (1 - R0) * c * c * c * c * c;   /* réflectance */
	double Tr = 1 - Re;                              /* transmittance */
	
	/* au-dela d'une certaine profondeur, on choisit aléatoirement si
	   on calcule le rayon réfléchi ou bien le rayon réfracté. En dessous du
	   seuil, on calcule les deux. */
	double rec[3];
	if (depth > SPLIT_DEPTH) {
		double P = .25 + .5 * Re;             /* probabilité de réflection */
		if (erand48(PRNG_state) < P) {
			radiance(x, reflected_dir, depth, PRNG_state, rec);
			double RP = Re / P;
			scal(RP, rec);
		} else {
			radiance(x, tdir, depth, PRNG_state, rec);
			double TP = Tr / (1 - P); 
			scal(TP, rec);
		}
	} else {
		double rec_re[3], rec_tr[3];
		radiance(x, reflected_dir, depth, PRNG_state, rec_re);
		radiance(x, tdir, depth, PRNG_state, rec_tr);
		zero(rec);
		axpy(Re, rec_re, rec);
		axpy(Tr, rec_tr, rec);
	}
	/* pondère, prend en compte la luminance */
	mul(f, rec, out);
	axpy(1, obj->emission, out);
	return;
}

double wtime()
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
	return (double)ts.tv_sec + ts.tv_usec / 1E6;
}

int toInt(double x)
{
	return pow(x, 1 / 2.2) * 255 + .5;   /* gamma correction = 2.2 */
} 

/* ######################################################################################################### */

void traiterPixel(const long pixel_num, const int w, const int h, unsigned short *const PRNG_state, const double * const camera_direction, const double * const camera_position, const double * const cx, const double * const cy , double *const image, const int samples){

	int i = pixel_num/w;
	int j = pixel_num%w;

	/* calcule la luminance d'un pixel, avec sur-échantillonnage 2x2 */
	double pixel_radiance[3] = {0, 0, 0};
	for (int sub_i = 0; sub_i < 2; sub_i++) {
		for (int sub_j = 0; sub_j < 2; sub_j++) {
			double subpixel_radiance[3] = {0, 0, 0};
			/* simulation de monte-carlo : on effectue plein de lancers de rayons et on moyenne */
			for (int s = 0; s < samples; s++) { 
				/* tire un rayon aléatoire dans une zone de la caméra qui correspond à peu près au pixel à calculer */
				double r1 = 2 * erand48(PRNG_state);
				double dx = (r1 < 1) ? sqrt(r1) - 1 : 1 - sqrt(2 - r1); 
				double r2 = 2 * erand48(PRNG_state);
				double dy = (r2 < 1) ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
				double ray_direction[3];
				copy(camera_direction, ray_direction);
				axpy(((sub_i + .5 + dy) / 2 + i) / h - .5, cy, ray_direction);
				axpy(((sub_j + .5 + dx) / 2 + j) / w - .5, cx, ray_direction);
				normalize(ray_direction);

				double ray_origin[3];
				copy(camera_position, ray_origin);
				axpy(140, ray_direction, ray_origin);						
				/* estime la lumiance qui arrive sur la caméra par ce rayon */
				double sample_radiance[3];
				radiance(ray_origin, ray_direction, 0, PRNG_state, sample_radiance);
				/* fait la moyenne sur tous les rayons */
				axpy(1. / samples, sample_radiance, subpixel_radiance);
			}

			clamp(subpixel_radiance);
			/* fait la moyenne sur les 4 sous-pixels */
			axpy(0.25, subpixel_radiance, pixel_radiance);
		}
	}
	
	copy(pixel_radiance, image + 3 * ((h - 1 - i) * w + j)); // <-- retournement vertical			
}


int traiterMsg(long *const debut_fin, MPI_Status * const status, const int my_rank, const int p, bool *const a_linitiative , bool *const exit_bool, bool *const travail_fini, bool *const demande, MPI_Request *const request){
	int demandeur = -2;
	switch((*status).MPI_TAG){
		case 1 : // qqun veut du travail 
			// on recupere qui veut du travail 
			MPI_Recv(&demandeur,1,MPI_INT,(*status).MPI_SOURCE, (*status).MPI_TAG,MPI_COMM_WORLD, status);
			//fprintf(stderr, "Processus %d | A recue de %d que %d  veut du travail et il lui reste de %ld à %ld à faire \n", my_rank, (my_rank-1), demandeur,  debut_fin[0], debut_fin[1] );
			if(demandeur == my_rank){
				// ma demande de travail m'est revenue , je notifie que je suis a l'initiative du msg de terminaison 
				*a_linitiative = true;
				// j'envoie un msg de terminaison, Tag = 3 !! 
				MPI_Isend(NULL,0,MPI_UNSIGNED_CHAR, (my_rank+1)%p , 3, MPI_COMM_WORLD, request);
				//fprintf(stderr, "Processus %d | Initie le stop \n", my_rank);
				return 4; // j'ai initié la demande de terminaison 
			}

			else { 

				#pragma omp critical (unique) 
				{
					if((debut_fin[1] - debut_fin[0]) >= 2 ){
						// jai du boulot, je lui envoie et je me met a jour et je met a jour ce que je dois faire ou retourner dans cette foncion
						// on envoie du travail : Tag 2
						//////////////// a faire en asynchrone  ? y aura t-il un gain ?
						long debut_fin_a_envoyer[2] = {debut_fin[0] + (debut_fin[1] - debut_fin[0])/2, debut_fin[1]};
						MPI_Send(debut_fin_a_envoyer,2,MPI_LONG, demandeur , 2, MPI_COMM_WORLD);
						debut_fin[1] = debut_fin[0] + (debut_fin[1] - debut_fin[0])/2;
						//fprintf(stderr, "Processus %d | Envoie à %d les pixels %ld à %ld et il lui en reste %ld \n", my_rank, demandeur, debut_fin_a_envoyer[0], debut_fin_a_envoyer[1] ,debut_fin[1] - debut_fin[0]); 
					}
					else{
						// jai pas de taff et j'envoie au suivant, je relai la demande quoi 
						//////////////// a faire en asynchrone  ? y aura t-il un gain ?
						MPI_Send(&demandeur,1,MPI_INT, (my_rank+1)%p , 1, MPI_COMM_WORLD);
						//fprintf(stderr, "Processus %d | Relai à %d que le processus %d veut du travail \n", my_rank, (my_rank+1)%p,demandeur);

					}
						
				}

				return 99;
			}
	
		case 2 : // on m'a envoye du boulooooot
			// je le recois 
			#pragma omp critical (unique)
			{
				MPI_Recv(debut_fin, 2, MPI_LONG, (*status).MPI_SOURCE, (*status).MPI_TAG,MPI_COMM_WORLD, status);
			}

			//fprintf(stderr, "Processus %d | A recu du travail de %d  \n", my_rank, (my_rank-1));

			*travail_fini = false;
			*demande = false;
			return 6; // j'ai recu du travail ! 

		default : // on me dit d'arreter, je recois STOP
			if(*a_linitiative){
				//fprintf(stderr, "Processus %d | A recu stop de %d et il était a a_linitiative d'un stop \n", my_rank, (my_rank-1));
				// je suis a l'initiative d'un stop donc je marrete la je ne relaie rien je sors
				*exit_bool = true;
				return 5; // terminaison !! 
			}

			else{
				// Je suis pas a l'initiative d'un STOP donc je relaie et je marrete ! 
				MPI_Send(NULL,0,MPI_UNSIGNED_CHAR, (my_rank+1)%p , 3, MPI_COMM_WORLD);

				//fprintf(stderr, "Processus %d | A recu stop de %d et il était paas a_linitiative d'un stop \n", my_rank, (my_rank-1));

				*exit_bool = true;
					
				return 5; // terminaison !! 
			}
	}
}

/* ################################################################################################################ */

int main(int argc, char **argv)
{ 

	/* Variables pour MPI */
  	/* rang du processus */
  	int my_rank;
  	/* nombre de processus */
  	int p;
  	int provided;

  	/* Initialisation de MPI */
  	MPI_Init_thread(&argc,&argv, MPI_THREAD_FUNNELED, &provided);
  	MPI_Comm_size(MPI_COMM_WORLD, &p);
  	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  	MPI_Status status;
  	MPI_Request request;
	int flag = -1;

	/* Petit cas test (small, quick and dirty): */
	int w = 600;
	int h = 500;
	int samples = 200;

	/* Gros cas test (big, slow and pretty): */
	/* int w = 3840; */
	/* int h = 2160; */
	/* int samples = 5000;  */

	if (argc == 2) 
		samples = atoi(argv[1]) / 4;

	static const double CST = 0.5135;  /* ceci défini l'angle de vue */
	double camera_position[3] = {50, 52, 295.6};
	double camera_direction[3] = {0, -0.042612, -1};
	normalize(camera_direction);

	/* incréments pour passer d'un pixel à l'autre */
	double cx[3] = {w * CST / h, 0, 0};    
	double cy[3];
	cross(cx, camera_direction, cy);  /* cy est orthogonal à cx ET à la direction dans laquelle regarde la caméra */
	normalize(cy);
	scal(CST, cy);

	/* précalcule la norme infinie des couleurs */
	int n = sizeof(spheres) / sizeof(struct Sphere);
	for (int i = 0; i < n; i++) {
		double *f = spheres[i].color;
		if ((f[0] > f[1]) && (f[0] > f[2]))
			spheres[i].max_reflexivity = f[0]; 
		else {
			if (f[1] > f[2])
				spheres[i].max_reflexivity = f[1];
			else
				spheres[i].max_reflexivity = f[2]; 
		}
	}

	/* ######################################################################################################################## */

	
	double debut = wtime();

	/* boucle principale */
	double *image = malloc(3 * w * h * sizeof(*image));
	if (image == NULL) {
		perror("Impossible d'allouer l'image");
		exit(1);
	}

	memset(image, 0x00, sizeof(double)*3 * w * h);

	long debut_fin[2] = {my_rank*((h*w)/p), (my_rank < p-1) ? (my_rank+1)*((h*w)/p) : w*h};
	bool exit_bool = false;
	bool travail_fini = false;

	#pragma omp parallel 
	{

		#pragma omp master 
		{
			bool a_linitiative = false;
			bool demande = false;

			
			while(!exit_bool){
				#pragma omp flush (travail_fini)
				MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
				if(flag == 1){
					traiterMsg(debut_fin, &status, my_rank, p, &a_linitiative, &exit_bool, &travail_fini, &demande, &request);
					#pragma omp flush (exit_bool ,travail_fini)
				}
				else if(travail_fini && !demande){
					MPI_Isend(&my_rank,1,MPI_INT, (my_rank+1)%p , 1, MPI_COMM_WORLD, &request);
					demande = true;
					//fprintf(stderr, "Processus %d | Envoie à %d que il veut du travail car il lui reste %ld \n", my_rank, (my_rank+1)%p, debut_fin[1] - debut_fin[0]);
				}
				else{
					continue;
				}
			}

			//fprintf(stderr, "temps passe a faire des comms %lf secondes \n",tps_comms);

		}

		long pix;
		unsigned short PRNG_state[3] = {0, 0, my_rank*my_rank*my_rank};
		bool inf_fin = false;
		double tps_calcul_effectif=0, debut_eff, fin_eff;

		while(!exit_bool){

			#pragma omp critical (unique)
			{
				pix = debut_fin[0]++;
				inf_fin = pix < debut_fin[1];
			}

			#pragma omp flush (exit_bool)

			if(inf_fin){
				debut_eff = wtime();
				traiterPixel(pix , w, h , PRNG_state , camera_direction , camera_position , cx, cy , image,samples);
				fin_eff = wtime();
				tps_calcul_effectif += fin_eff - debut_eff;
				//fprintf(stderr, "Processus %d | fait le pixel %ld  \n", my_rank, pix);
				#pragma omp flush (exit_bool)
			}
			else if (!exit_bool){

				#pragma omp flush (travail_fini, exit_bool)
				if(!travail_fini){
					travail_fini = true; 
					#pragma omp flush (travail_fini, exit_bool)
				}
				
				while(!exit_bool && travail_fini){
					//fprintf(stderr, "Processus %d | bloque \n", my_rank, pix);
					#pragma omp flush (travail_fini, exit_bool)
				}
			}
			else{continue;}

		}

		fprintf(stderr, "temps calcul effectif %lf secondes \n",tps_calcul_effectif);

	}

	double *image_final = NULL ;

	/////////////////// Tester un autre processus qui ramasse et ecrit le fichier ? 
	if(my_rank == 0 ){

		image_final = malloc(3 * w * h * sizeof(*image_final));

		if (image_final == NULL) {
			perror("Impossible d'allouer l'image_final");
			exit(1);
		}
	}

    /////////////////// Tester un autre processus qui ramasse et ecrit le fichier ? 
	MPI_Reduce(image, image_final, 3 * w * h , MPI_DOUBLE ,MPI_SUM, 0 , MPI_COMM_WORLD);

	double fin = wtime();

	/* stocke l'image dans un fichier au format NetPbm */
	if(my_rank == 0 ) {

		fprintf(stderr, "temps total :  %lf secondes \n", fin-debut);

		struct passwd *pass; 
		char nom_sortie[100] = "";
		char nom_rep[30] = "";

		pass = getpwuid(getuid()); 
		sprintf(nom_rep, "/tmp/%s", pass->pw_name);
		mkdir(nom_rep, S_IRWXU);
		sprintf(nom_sortie, "%s/image.ppm", nom_rep);
		
		FILE *f = fopen(nom_sortie, "w");
		fprintf(f, "P3\n%d %d\n%d\n", w, h, 255); 

		for (int i = 0; i < w * h; i++) 
	  		fprintf(f,"%d %d %d ", toInt(image_final[3 * i]), toInt(image_final[3 * i + 1]), toInt(image_final[3 * i + 2])); 
		fclose(f); 

		free(image_final);
	}

	free(image);

	MPI_Finalize();
}
