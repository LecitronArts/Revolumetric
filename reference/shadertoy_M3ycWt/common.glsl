//CONST
const float PI = 3.141592653;
const float I3 = 1./3.;
const float I9 = 1./9.;
const float I12 = 1./12.;
const float I32 = 1./32.;
const float I48 = 1./48.;
const float I256 = 1./256.;
const float I512 = 1./512.;
const float I1024 = 1./1024.;
const float SUN_DIST = 48.;
const float SUN_SM_SIZE = 32.;
const float ISUNRT = 2.*PI/16.;
const float SUNRTO = 1.15;

//DEFINE
#define RES iChannelResolution[0].xy
#define IRES 1./iChannelResolution[0].xy
#define ASPECT vec2(iChannelResolution[0].x/iChannelResolution[0].y,1.)

//MATH
vec3 BRDF_GGX(vec3 w_o, vec3 w_i, vec3 n, float alpha, vec3 F0) {
    vec3 h = normalize(w_i + w_o);
    float a2 = alpha*alpha;
    float D = a2/(3.141592653*pow(pow(dot(h, n), 2.)*(a2 - 1.) + 1., 2.));
    vec3 F = F0 + (1. - F0)*pow(1. - dot(n, w_o), 5.);
    float k = a2*0.5;
    float G = 1./((dot(n, w_i)*(1. - k) + k)*(dot(n, w_o)*(1. - k) + k));
    vec3 OUT = F*(D*G*0.25);
    return ((isnan(OUT) != bvec3(false)) ? vec3(0.) : OUT);
}

mat3 TBN(vec3 N) {
    vec3 Nb, Nt;
    if (abs(N.y) > 0.999) {
        Nb = vec3(1., 0., 0.);
        Nt = vec3(0., 0., 1.);
    } else {
    	Nb = normalize(cross(N, vec3(0., 1., 0.)));
    	Nt = normalize(cross(Nb, N));
    }
    return mat3(Nb.x, Nt.x, N.x, Nb.y, Nt.y, N.y, Nb.z, Nt.z, N.z);
}

float smin(float a, float b, float k) {
    //https://iquilezles.org/articles/smin
    float h = max(k-abs(a-b),0.)/k;
    return min(a,b)-h*h*h*k*(1.0/6.0);
}

float DFBox(vec2 p, vec2 b) {
    vec2 d = abs(p - b*0.5) - b*0.5;
    return min(max(d.x, d.y), 0.) + length(max(d, 0.));
}

float DFBox(vec3 p, vec3 b) {
    vec3 d = abs(p - b*0.5) - b*0.5;
    return min(max(d.x, max(d.y, d.z)), 0.) + length(max(d, 0.));
}

vec2 ABox(vec3 origin, vec3 idir, vec3 bmin, vec3 bmax) {
    vec3 tMin = (bmin - origin)*idir;
    vec3 tMax = (bmax - origin)*idir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    return vec2(max(max(t1.x, t1.y), t1.z), min(min(t2.x, t2.y), t2.z));
}

vec3 ABoxfarNormal(vec3 origin, vec3 idir, vec3 signdir, vec3 bmin, vec3 bmax, out float dist) {
    vec3 tMin = (bmin - origin)*idir;
    vec3 tMax = (bmax - origin)*idir;
    vec3 t2 = max(tMin, tMax);
    dist = min(min(t2.x, t2.y), t2.z);
    if (t2.x < min(t2.y, t2.z)) return vec3(signdir.x, 0., 0.);
    else if (t2.y < t2.z) return vec3(0., signdir.y, 0.);
    else return vec3(0., 0., signdir.z);
}

//SCENE
vec3 GetSkyLight(vec3 d, float t) {
    return vec3(0.75, 0.85, 1.)*1.25*pow(d.y*0.5 + 0.5, 2.);
}

vec3 GetSunLight(float t) {
    return vec3(2., 1.5, 1.25);
}

vec3 GetSunDir(float t) {
    float aniT = t*(1. - exp(-t*0.2));
    float sunA = aniT*ISUNRT + SUNRTO;
    float sunT = (pow(cos(aniT*ISUNRT*0.25), 5.)*0.5 + 0.5)*2. + 0.2;
    return normalize(vec3(sin(sunA), sunT, cos(sunA)));
}

//HEMISPHERE
vec3 ComputeDirEven(vec2 uv, float probeSize) {
    vec2 probeRel = uv - probeSize*0.5;
    float probeThetai = max(abs(probeRel.x), abs(probeRel.y));
    float probeTheta = probeThetai/probeSize*3.14192653;
    float probePhi = 0.;
    if (probeRel.x + 0.5 > probeThetai && probeRel.y - 0.5 > -probeThetai) {
        probePhi = probeRel.x - probeRel.y;
    } else if (probeRel.y - 0.5 < -probeThetai && probeRel.x - 0.5 > -probeThetai) {
        probePhi = probeThetai*2. - probeRel.y - probeRel.x;
    } else if (probeRel.x - 0.5 < -probeThetai && probeRel.y + 0.5 < probeThetai) {
        probePhi = probeThetai*4. - probeRel.x + probeRel.y;
    } else if (probeRel.y + 0.5 > probeThetai && probeRel.x + 0.5 < probeThetai) {
        probePhi = probeThetai*8. - (probeRel.y - probeRel.x);
    }
    probePhi = probePhi*3.141592653*2./(4. + 8.*floor(probeThetai));
    return vec3(vec2(sin(probePhi), cos(probePhi))*sin(probeTheta), cos(probeTheta));
}

vec3 ComputeDir(vec2 uv, float probeSize) {
    if (probeSize > 4.5) return ComputeDirEven(uv, probeSize);
    vec2 probeRel = uv - 1.5;
    if (length(probeRel) < 0.1) return vec3(0., 0., 1.);
    float probePhi = atan(probeRel.x, probeRel.y) + 3.141592653*1.75;
    float probeTheta = 3.141592653*0.25;
    return vec3(vec2(sin(probePhi), cos(probePhi))*sin(probeTheta), cos(probeTheta));
}

vec2 ProjectDir(vec3 dir, float probeSize) {
    if (dir.z <= 0.) return vec2(-1.);
    float thetai = min(floor((1. - acos(length(dir.xy)/length(dir))/(3.141592653*0.5))*(probeSize*0.5)), probeSize*0.5 - 1.);
    float phiF = atan(-dir.x, -dir.y);
    float phiI = floor((phiF/3.141592653*0.5 + 0.5)*(4. + 8.*thetai) + 0.5) + 0.5;
    vec2 phiUV;
    float phiLen = 2.*thetai + 1.;
    float sideLen = phiLen + 1.;
    if (phiI < phiLen) phiUV = vec2(sideLen - 0.5, sideLen - phiI);
    else if (phiI < phiLen*2.) phiUV = vec2(sideLen - (phiI - phiLen), 0.5);
    else if (phiI < phiLen*3.) phiUV = vec2(0.5, phiI - phiLen*2.);
    else phiUV = vec2(phiI - phiLen*3., sideLen - 0.5);
    return vec2((probeSize - sideLen)*0.5) + phiUV;
}