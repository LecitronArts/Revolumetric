/*
A volumetric RC implementation
    Scene
        Volume resolution is 32X32X48, with 6 hemispheres per voxel
        Everything is dynamic, temporal merging means some light lag is inevitable
    Rays
        A hemisphere traces (3x3)*(4^N) rays, where N is the LOD <- (0, 1, 2, 3, ..)
    Probe placement
        Must decide where we trace rays from inside the voxel
        Very simple implementation in this shader -> choose the first empty voxel near the center if it exists
            A better approach is beyond this shader :)
    Merging
        Weighted trilinear interpolation is used
        Visibility determines the weight
            Is computed by projecting lower cascades on the higher cascade probes to determine visibility
            Uses the nearest hemisphere and its traced rays -> one texture fetch for visibility
    Performance
        Rays with almost the same direction are traced close to each other in the cubemap
        Use acceleration structure when tracing rays
    These are examples of other improvements:
        Probe placement inside voxels
            Find probe position using volume mipmaps
        Overlap between 6 hemispheres in one voxel
            A lot of rays cover the same directions from the same origin -> bad
            But don't want to lose cos(theta) in the integral if rays are shared between hemispheres
                Could store multiple weights per rays for the 3 closest hemispheres (max 3 hemisphere overlap)
            Other implementations might have solved this ..



Controls:
    Animation stops when the mouse is used
        WASD - move the camera
        Space - move faster
        Mouse - rotate the camera
*/


vec4 TextureCube(vec2 uv) {
    float tcSign = -mod(floor(uv.y*I1024), 2.)*2. + 1.;
    vec3 tcD = vec3(vec2(uv.x, mod(uv.y, 1024.))*I512 - 1., tcSign);
    if (uv.y > 4096.) tcD = tcD.xzy;
    else if (uv.y > 2048.) tcD = tcD.zxy;
    return textureLod(iChannel3, tcD, 0.);
}

vec3 IntegrateVoxel(vec3 p, vec3 n) {
    vec3 l = vec3(0.);
    vec2 cubeUV = vec2(p.x + floor(p.y)*32., p.z);
    float normalOffset;
    vec3 an = abs(n);
    if (an.x > max(an.y, an.z)) normalOffset = ((n.x < 0.)? 0. : 48.)*9.;
    else if (an.y > an.z) normalOffset = ((n.y < 0.)? 48.*2. : 48.*3.)*9.;
    else normalOffset = ((n.z < 0.)? 48.*4. : 48.*5.)*9.;
    for (float i = 0.; i < 8.5; i++) {
        l += TextureCube(cubeUV + vec2(0., normalOffset + i*48.)).xyz;
    }
    return l;
}

float SampleDotShadow(vec3 p, vec3 n) {
    vec3 SUN_DIR = GetSunDir(iTime);
    vec3 SUN_TAN = normalize(cross(SUN_DIR, vec3(0., 1., 0.)));
    vec3 SUN_BIT = cross(SUN_TAN, SUN_DIR);
    vec3 an = abs(n);
    vec3 nt, nb;
    if (an.x > max(an.y, an.z)) {
        nt = vec3(0., 0., 1.);
        nb = vec3(0., 1., 0.);
    } else if (an.y > an.z) {
        nt = vec3(1., 0., 0.);
        nb = vec3(0., 0., 1.);
    } else {
        nt = vec3(1., 0., 0.);
        nb = vec3(0., 1., 0.);
    }
    float o = 0.;
    for (float x = -0.15; x < 0.2; x += 0.15) {
        for (float y = -0.15; y < 0.2; y += 0.15) {
            vec3 sp = p + nt*x + nb*y;
            vec3 smPos = sp - (vec3(16., 16., 24.) + SUN_DIR*SUN_DIST);
            vec2 smUV = vec2(dot(smPos, SUN_TAN), dot(smPos, SUN_BIT))/SUN_SM_SIZE/ASPECT*0.5 + 0.5;
            if (texture(iChannel2, smUV).w > dot(smPos, -SUN_DIR)) o += 1.;
        }
    }
    return max(0., dot(SUN_DIR, n))*o/9.;
}

vec4 Trace(vec3 p, vec3 d, out vec3 vp, out vec4 vc) {
    vec4 info = vec4(vec3(0.), 100000.);
    vec3 signdir = (max(vec3(0.), sign(d))*2. - 1.);
    vec3 iDir = 1./d;
    float bbDF = DFBox(p, vec3(32., 32., 48.));
    vec2 bb = ABox(p, iDir, vec3(0.01), vec3(31.99, 31.99, 47.99));
    if (bbDF > 0. && (bb.x < 0. || bb.y < bb.x)) return vec4(-10.);
    float tFAR = bb.y;
    float t = ((bbDF < 0.)? 0. : bb.x + 0.001);
    vec3 cp;
    vec4 sC;
    vec3 fp = floor(p + d*t);
    vec3 lfp = fp - vec3(0., 1., 0.);
    for (int i = 0; i < 128; i++) {
        if (t > tFAR) break;
        cp = p + d*t;
        sC = texture(iChannel1, vec2(fp.x + (mod(fp.y + 0.5, 8.) - 0.5)*32. + 0.5, fp.z + floor(fp.y*0.125)*48. + 0.5)*IRES);
        if (sC.w > 0.5) {
            vp = lfp + 0.5;
            vc = sC;
            return vec4(lfp - fp, t);
        }
        lfp = fp;
        fp += ABoxfarNormal(p, iDir, signdir, fp, fp + 1., t);
    }
    return vec4(-10.);
}

vec3 AcesFilm(vec3 x) {
    return clamp((x*(2.51*x + 0.03))/(x*(2.43*x + 0.59) + 0.14), 0., 1.);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec3 color = vec3(0.);
    vec3 cameraPos = texture(iChannel0, vec2(3.5, 0.5)*IRES).xyz;
    vec3 cameraEye = texture(iChannel0, vec2(2.5, 0.5)*IRES).xyz;
    vec3 pDir = normalize(vec3((fragCoord*IRES*2. - 1.)*ASPECT, 1.)*TBN(cameraEye));
    vec3 pPos;
    vec4 pCol;
    vec4 pHit = Trace(cameraPos, pDir, pPos, pCol);
    if (pHit.x > -1.5) {
        if (pCol.w > 1.5) {
            color = pCol.xyz;
        } else {
            color += IntegrateVoxel(pPos, pHit.xyz);
            color += SampleDotShadow(cameraPos + pDir*pHit.w + pHit.xyz*0.25, pHit.xyz)*GetSunLight(iTime);
            color *= pCol.xyz;
        }
    } else {
        color += GetSkyLight(pDir, iTime);
    }
    fragColor = vec4(pow(AcesFilm(max(vec3(0.), color)), vec3(0.45)), 1.);
}