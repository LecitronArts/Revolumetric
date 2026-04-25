//Settings
const float FOV=0.9; //In radians
const float VoxelSize=0.125;
//Constants
const float I16=1./16.;
const float I32=1./32.;
const float I64=1./64.;
const float I512=1./512.;
const float PI=3.14159265;
const vec2 eps=vec2(0.001,0.);
const float CFOV=tan(FOV);
#define IRES 1./iResolution.xy
#define ASPECT vec2(iResolution.x/iResolution.y,1.)
//Structs
struct DF { float D; vec3 C; float Mat; }; //Mat 0-1 Mirror-Diffuse, 2=Emissive
struct HIT { float D; vec3 P; vec3 N; vec3 C; float Mat; };
vec3 ConeDirections[16]=vec3[16](
 vec3(0.5773502691896258,0.5773502691896258,0.5773502691896258)
,vec3(0.5773502691896258,-0.5773502691896258,-0.5773502691896258)
,vec3(-0.5773502691896258,0.5773502691896258,-0.5773502691896258)
,vec3(-0.5773502691896258,-0.5773502691896258,0.5773502691896258)
,vec3(-0.9030073291598593,-0.1826964031330545,-0.3888441690006706)
,vec3(-0.9030073291598593,0.1826964031330545,0.3888441690006706)
,vec3(0.9030073291598593,-0.1826964031330545,0.3888441690006706)
,vec3(0.9030073291598593,0.1826964031330545,-0.3888441690006706)
,vec3(-0.3888441690006706,-0.9030073291598593,-0.1826964031330545)
,vec3(0.3888441690006706,-0.9030073291598593,0.1826964031330545)
,vec3(0.3888441690006706,0.9030073291598593,-0.1826964031330545)
,vec3(-0.3888441690006706,0.9030073291598593,0.1826964031330545)
,vec3(-0.1826964031330545,-0.3888441690006706,-0.9030073291598593)
,vec3(0.1826964031330545,0.3888441690006706,-0.9030073291598593)
,vec3(-0.1826964031330545,0.3888441690006706,0.9030073291598593)
,vec3(0.1826964031330545,-0.3888441690006706,0.9030073291598593));


float Box(vec3 p, vec3 b) {
    vec3 d=abs(p-b*0.5)-b*0.5;
    return min(max(d.x,max(d.y,d.z)),0.)+length(max(d,0.));
}

float BoxC(vec3 p, vec3 b) {
    vec3 d=abs(p)-b;
    return min(max(d.x,max(d.y,d.z)),0.)+length(max(d,0.));
}

vec2 Rotate(vec2 p, float ang) {
    float c=cos(ang), s=sin(ang);
    return vec2(p.x*c-p.y*s,p.x*s+p.y*c);
}

float RotatedBox(vec3 p, vec3 b, float rz, float rx) {
    vec3 pp=p;
    pp.xy=Rotate(p.xy,rz);
    pp.yz=Rotate(pp.yz,rx);
    return BoxC(pp,b);
}

vec2 Repeat(vec2 p, float n) {
    float ang=2.*3.14159/n;
    float sector=floor(atan(p.x,p.y)/ang+0.5);
    p=Rotate(p,sector*ang);
    return p;
}

float Plane(vec3 p, vec3 n, float offs) {
	return dot(p,n)-offs;
}

float Line(vec3 p, vec3 a, vec3 b) {
    vec3 ba=b-a;
    float k=dot(p-a,ba)/dot(ba,ba);
    return length((a+clamp(k,0.,1.)*(b-a))-p);
}

float SMIN(float a, float b, float k) {
    float h=clamp(0.5+0.5*(b-a)/k,0.,1.);
    return mix(b,a,h)-k*h*(1.-h);
}

void MIN(inout DF df, DF a) {
    if (a.D<=df.D) df=a;
}

DF SDF(vec3 p, float Time) {
    DF df=DF(p.y-0.25,vec3(1.),1.);
    //Emissive plane
    vec3 ep=p; ep.yz=Rotate(ep.yz,Time*0.78*0.+0.78);
    MIN(df,DF(Box(p,vec3(0.125,5.,8.)),vec3(1.+mod(floor(ep.y)+floor(ep.z),2.)*2.)*vec3(0.4,1.,0.4)*1.5
              ,mod(floor(ep.y)+floor(ep.z),2.)+1.));

    //Diffuse plane
    MIN(df,DF(Box(p-vec3(0.,0.,7.75),vec3(8.,5.,0.25)),vec3(1.),1.));
    	MIN(df,DF(Box(p-vec3(2.5,2.,6.),vec3(2.,2.,2.)),vec3(1.),1.)); //Box på vägg
    //Låg roterande emissive object
    vec3 rp=p; rp.xz=Rotate(rp.xz-vec2(4.,4.),Time*0.75);
    MIN(df,DF(max(BoxC(rp,vec3(2.,1.,2.)),-BoxC(rp,vec3(1.5,8.,1.5))),vec3(1.,0.25,0.2)*4.,2.));

    //Modulation
    vec3 mp=p*12.;
    df.D+=(sin(mp.x)*sin(mp.y)*sin(mp.z))*0.015;

    //Spheres
    MIN(df,DF(length(vec3(fract(p.x)-0.5,p.y-0.55,fract(p.z)-0.5))-0.3,vec3(1.),
             ((mod(floor(p.x)+floor(p.z),2.)==1.)?1.:0.2)));
    //Glossy plane
    MIN(df,DF(Box(p,vec3(8.,5.,0.25)),vec3(1.),((mod(floor(p.y*1.)+floor(p.x*1.),2.)==1.)?0.1:0.25)));

    //Return
	return df;
}

vec3 Gradient(vec3 p, float t) {
    return normalize(vec3(
        SDF(p+eps.xyy,t).D-SDF(p-eps.xyy,t).D,
        SDF(p+eps.yxy,t).D-SDF(p-eps.yxy,t).D,
        SDF(p+eps.yyx,t).D-SDF(p-eps.yyx,t).D));
}

mat3 TBN(vec3 N) {
    vec3 Nb,Nt;
    if (abs(N.y)>0.999) {
        Nb=vec3(1.,0.,0.);
        Nt=vec3(0.,0.,1.);
    } else {
    	Nb=normalize(cross(N,vec3(0.,1.,0.)));
    	Nt=normalize(cross(Nb,N));
    }
    return mat3(Nb.x,Nt.x,N.x,Nb.y,Nt.y,N.y,Nb.z,Nt.z,N.z);
}

vec3 TBN(vec3 N, out vec3 O) {
    O=normalize(cross(N,vec3(0.,1.,0.)));
    return normalize(cross(O,N));
}

float boxfar(vec3 origin, vec3 dir, vec3 bmin, vec3 bmax) {
    vec3 tMin=(bmin-origin)*dir;
    vec3 tMax=(bmax-origin)*dir;
    vec3 t2=max(tMin,tMax);
    return min(min(t2.x,t2.y),t2.z);
}

vec2 box(vec3 origin, vec3 dir, vec3 bmin, vec3 bmax) {
    vec3 tMin=(bmin-origin)*dir;
    vec3 tMax=(bmax-origin)*dir;
    vec3 t1=min(tMin,tMax);
    vec3 t2=max(tMin,tMax);
    return vec2(max(max(t1.x,t1.y),t1.z),min(min(t2.x,t2.y),t2.z));
}

float boxfar2(vec2 origin, vec2 dir, vec2 bmin, vec2 bmax) {
    vec2 tMin=(bmin-origin)*dir;
    vec2 tMax=(bmax-origin)*dir;
    vec2 t2=max(tMin,tMax);
    return min(t2.x,t2.y);
}

bool TraceRay(vec3 pos, vec3 dir, out HIT R, float Time) {
    DF t; float dist=0.; R.P=pos;
    float FAR;
    if (pos.x>0. && pos.x<8. && pos.z>0. && pos.z<8.) {
    	FAR=boxfar(pos,1./dir,vec3(0.),vec3(8.));
    } else {
        vec2 BB=box(pos,1./dir,vec3(0.),vec3(8.));
        if (BB.x>0. && BB.y>BB.x) {
            FAR=BB.y;
            dist=BB.x;
        } else
            return false;
    }
    for (int i=0; i<128; i++) {
        if (dist>FAR) break;
        R.P=pos+dir*dist;
        t=SDF(R.P,Time);
        if (t.D<eps.x) {
			R.D=dist;
            R.N=Gradient(R.P,Time);
            R.C=t.C;
            R.Mat=t.Mat;
            return true;
        }
        dist=dist+t.D;
    }
	return false;
}
