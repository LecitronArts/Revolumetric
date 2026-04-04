//Radiance cascades

const float LOD_MAX_POS   = 5022.;
const float LOD_POS[5]    = float[5](0., 2592., 3888., 4536., 4860.);

vec4 TextureCube(vec2 uv) {
    float tcSign = -mod(floor(uv.y*I1024), 2.)*2. + 1.;
    vec3 tcD = vec3(vec2(uv.x, mod(uv.y, 1024.))*I512 - 1., tcSign);
    if (uv.y > 4096.) tcD = tcD.xzy;
    else if (uv.y > 2048.) tcD = tcD.zxy;
    return textureLod(iChannel3, tcD, 0.);
}

vec4 TextureCube(vec2 uv, float lod) {
    float tcSign = -mod(floor(uv.y*I1024), 2.)*2. + 1.;
    vec3 tcD = vec3(vec2(uv.x, mod(uv.y, 1024.))*I512 - 1., tcSign);
    if (uv.y > 4096.) tcD = tcD.xzy;
    else if (uv.y > 2048.) tcD = tcD.zxy;
    return textureLod(iChannel3, tcD, lod);
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

float SampleShadow(vec3 p, vec3 n, vec3 SUN_DIR) {
    vec3 SUN_TAN = normalize(cross(SUN_DIR, vec3(0., 1., 0.)));
    vec3 SUN_BIT = cross(SUN_TAN, SUN_DIR);
    vec3 smPos = p - (vec3(16., 16., 24.) + SUN_DIR*SUN_DIST);
    vec2 smUV = vec2(dot(smPos, SUN_TAN), dot(smPos, SUN_BIT))/SUN_SM_SIZE/ASPECT*0.5 + 0.5;
    return float(texture(iChannel2, smUV).w > dot(smPos, -SUN_DIR));
}

vec4 NearestCS(vec3 fPos, vec2 lFaceUV, vec3 lSize, vec4 lDIUV0, vec4 lDIUV1,
               vec3 vPos, float lLF, float lOff, float lPS, float lPow4) {
    vec2 lVolumeUV = vec2(fPos.x + fPos.y*lSize.x + 0.5, fPos.z + 0.5);
    vec3 rlPos = (fPos + 0.5)*lLF;
    vec3 rPos = vPos - rlPos;
    vec3 aPos = abs(rPos);
    vec2 oFaceUV = vec2(0., lOff);
    if (aPos.x > max(aPos.y, aPos.z)) {
        oFaceUV.y += ((rPos.x < 0.)? 0. : 1.)*9.*lSize.z;
        rPos = vec3(rPos.zy, aPos.x);
    } else if (aPos.y > aPos.z) {
        oFaceUV.y += ((rPos.y < 0.)? 2. : 3.)*9.*lSize.z;
        rPos = vec3(rPos.xz, aPos.y);
    } else {
        oFaceUV.y += ((rPos.z < 0.)? 4. : 5.)*9.*lSize.z;
        rPos = vec3(rPos.xy, aPos.z);
    }
    vec2 projUV = ProjectDir(rPos, lPS);
    float oDirIndex = projUV.x + floor(projUV.y)*lPS;
    vec2 oDirUV = vec2(floor(mod(oDirIndex, lPow4))*lSize.x*lSize.y, floor(oDirIndex/lPow4)*lSize.z);
    float projectedRayLen = TextureCube(oFaceUV + oDirUV + lVolumeUV).w;
    if (projectedRayLen < length(rPos) - 0.5) return vec4(0., 0., 0., 0.00001);
    vec4 lOut = TextureCube(lFaceUV + lDIUV0.xy + lVolumeUV) +
                TextureCube(lFaceUV + lDIUV0.zw + lVolumeUV) +
                TextureCube(lFaceUV + lDIUV1.xy + lVolumeUV) +
                TextureCube(lFaceUV + lDIUV1.zw + lVolumeUV);
    if (lOut.w < -0.5) return vec4(lOut.xyz*0.001, 0.001);
    return vec4(lOut.xyz, 1.);
}

vec4 TrilinearCS(vec3 lPos, vec2 lFaceUV, float lDirIndex, float lProbeSize,
                 vec3 lSize, float lPow4, vec3 vPos, float lLF, float lOff) {
    vec4 lDIUV0 = vec4(floor(mod(lDirIndex, lPow4))*lSize.x*lSize.y, floor(lDirIndex/lPow4)*lSize.z,
                       floor(mod(lDirIndex + 1., lPow4))*lSize.x*lSize.y, floor((lDirIndex + 1.)/lPow4)*lSize.z);
    vec4 lDIUV1 = vec4(floor(mod(lDirIndex + lProbeSize, lPow4))*lSize.x*lSize.y,
                       floor((lDirIndex + lProbeSize)/lPow4)*lSize.z,
                       floor(mod(lDirIndex + lProbeSize + 1., lPow4))*lSize.x*lSize.y,
                       floor((lDirIndex + lProbeSize + 1.)/lPow4)*lSize.z);
    vec3 fPos = clamp(floor(lPos - 0.5), vec3(0.), lSize - 2.);
    vec3 frPos = min(vec3(1.), lPos - 0.5 - fPos);
    vec4 l000 = NearestCS(fPos, lFaceUV, lSize, lDIUV0, lDIUV1, vPos, lLF, lOff, lProbeSize, lPow4);
    vec4 l100 = NearestCS(vec3(fPos.x + 1., fPos.y, fPos.z), lFaceUV,
                          lSize, lDIUV0, lDIUV1, vPos, lLF, lOff, lProbeSize, lPow4);
    vec4 l010 = NearestCS(vec3(fPos.x, fPos.y + 1., fPos.z), lFaceUV,
                          lSize, lDIUV0, lDIUV1, vPos, lLF, lOff, lProbeSize, lPow4);
    vec4 l110 = NearestCS(vec3(fPos.x + 1., fPos.y + 1., fPos.z), lFaceUV,
                          lSize, lDIUV0, lDIUV1, vPos, lLF, lOff, lProbeSize, lPow4);
    vec4 l001 = NearestCS(vec3(fPos.x, fPos.y, fPos.z + 1.), lFaceUV,
                          lSize, lDIUV0, lDIUV1, vPos, lLF, lOff, lProbeSize, lPow4);
    vec4 l101 = NearestCS(vec3(fPos.x + 1., fPos.y, fPos.z + 1.), lFaceUV,
                          lSize, lDIUV0, lDIUV1, vPos, lLF, lOff, lProbeSize, lPow4);
    vec4 l011 = NearestCS(vec3(fPos.x, fPos.y + 1., fPos.z + 1.), lFaceUV,
                          lSize, lDIUV0, lDIUV1, vPos, lLF, lOff, lProbeSize, lPow4);
    vec4 l111 = NearestCS(fPos + 1., lFaceUV, lSize, lDIUV0, lDIUV1, vPos, lLF, lOff, lProbeSize, lPow4);
    float lweight = mix(mix(mix(l000.w, l100.w, frPos.x), mix(l010.w, l110.w, frPos.x), frPos.y),
                    mix(mix(l001.w, l101.w, frPos.x), mix(l011.w, l111.w, frPos.x), frPos.y), frPos.z);
    return vec4(mix(mix(mix(l000.xyz, l100.xyz, frPos.x), mix(l010.xyz, l110.xyz, frPos.x), frPos.y),
                    mix(mix(l001.xyz, l101.xyz, frPos.x), mix(l011.xyz, l111.xyz, frPos.x), frPos.y),
                    frPos.z)/lweight, lweight);
}

bool OutsideGeo(vec3 sp) {
    vec3 p = floor(sp) + 0.5;
    return (texture(iChannel1, vec2(p.x + floor(mod(p.y, 8.))*32., p.z + floor(p.y*0.125)*48.)*IRES).w < 0.5);
}

vec3 GeoOffset(vec3 vertex) {
    for (float x = -0.5; x < 1.; x++) {
        for (float y = -0.5; y < 1.; y++) {
            for (float z = -0.5; z < 1.; z++) {
                if (OutsideGeo(vertex + vec3(x, y, z))) return vec3(x, y, z)*0.1;
            }
        }
    }
    return vec3(0.001);
}

vec4 Trace(vec3 p, vec3 d, out vec3 pPos, out vec4 pCol) {
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
            pPos = lfp + 0.5;
            pCol = sC;
            return vec4(lfp - fp, t);
        }
        lfp = fp;
        fp += ABoxfarNormal(p, iDir, signdir, fp, fp + 1., t);
    }
    return vec4(-10.);
}

void mainCubemap(out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir) {
    vec4 info = texture(iChannel3, rayDir);
    vec2 uv; vec3 aDir = abs(rayDir);
    if (aDir.z > max(aDir.x, aDir.y)) {
        //Z-side
        uv = floor(((rayDir.xy/aDir.z)*0.5 + 0.5)*1024.) + 0.5;
        if (rayDir.z < 0.) uv.y += 1024.;
    } else if (aDir.x > aDir.y) {
        //X-side
        uv = floor(((rayDir.yz/aDir.x)*0.5 + 0.5)*1024.) + 0.5;
        if (rayDir.x > 0.) uv.y += 2048.;
        else uv.y += 3072.;
    } else {
        //Y-side
        uv = floor(((rayDir.xz/aDir.y)*0.5 + 0.5)*1024.) + 0.5;
        if (rayDir.y > 0.) uv.y += 4096.;
        else uv.y += 5120.;
    }
    
    //Cascades
    if (uv.y < LOD_MAX_POS) {
        float vLod = 0.;
        vec2 vuv = uv;
        float lOffset = 2592.;
        for (int i = 1; i < 5; i++) {
            lOffset = LOD_POS[i];
            if (uv.y > lOffset) {
                vLod += 1.;
            } else {
                vuv.y -= LOD_POS[i - 1];
                break;
            }
        }
        float vLodFactor = pow(2., vLod);
        float vILodFactor = 1./vLodFactor;
        float vProbeSize = 3.*vLodFactor;
        vec3 vSize = vec3(32., 32., 48.)*vILodFactor;
        vec3 vISize = vec3(I32, I32, I48)*vLodFactor;
        float vFace = mod(floor(vuv.y*vISize.z*I9) + 0.5, 6.);
        float vDirIndex = floor(mod(vuv.y*vISize.z, 9.))*pow(4., vLod) + floor(vuv.x*vISize.x*vISize.y) + 0.5;
        vec3 vPos = vec3(mod(vuv.x, vSize.x), mod(floor(vuv.x*vISize.x) + 0.5, vSize.y), mod(vuv.y, vSize.z))*vLodFactor;
        vec3 vN, vT, vB;
        if (vFace < 2.) {
            vT = vec3(0., 0., 1.);
            vB = vec3(0., 1., 0.);
            vN = vec3(((vFace < 1.)? -1. : 1.), 0., 0.);
        } else if (vFace < 4.) {
            vT = vec3(1., 0., 0.);
            vB = vec3(0., 0., 1.);
            vN = vec3(0., ((vFace < 3.)? -1. : 1.), 0.);
        } else {
            vT = vec3(1., 0., 0.);
            vB = vec3(0., 1., 0.);
            vN = vec3(0., 0., ((vFace < 5.)? -1. : 1.));
        }
        if (vLod > 0.5) {
            vPos += GeoOffset(vPos);
        } else {
            vPos -= vN*0.25;
        }
        if (OutsideGeo(vPos)) {
            info = vec4(0., 0., 0., 1.);
            vec2 vDirUV = vec2(mod(vDirIndex, vProbeSize), floor(vDirIndex/vProbeSize) + 0.5);
            vec3 vDir = ComputeDir(vDirUV, vProbeSize);
            vDir = vT*vDir.x + vB*vDir.y + vN*vDir.z;
            vec3 pPos; vec4 pCol;
            vec4 vHit = Trace(vPos, vDir, pPos, pCol);
            if (vHit.x > -1.5) {
                info.w = vHit.w;
                if (pCol.w > 1.5) {
                    info.xyz = pCol.xyz;
                } else {
                    vec3 SUN_DIR = GetSunDir(iTime);
                    float sunDot = dot(SUN_DIR, vHit.xyz);
                    if (sunDot > 0.) info.xyz += SampleShadow(pPos, vHit.xyz, SUN_DIR)*sunDot*GetSunLight(iTime);
                    info.xyz += IntegrateVoxel(floor(pPos - vDir*0.001) + 0.5, vHit.xyz);
                    info.xyz *= pCol.xyz;
                }
            } else {
                info.w = 10000000.;
                info.xyz += GetSkyLight(vDir, iTime);
            }
            
            //Normalized area * cos(theta)
            if (vLod < 0.5) {
                if (length(vDirUV) > 0.75) {
                    info.xyz *= cos(0.25*3.141592653)/8.;
                } else {
                    info.xyz *= (1. - cos(0.25*3.141592653));
                }
            } else {
                vec2 vProbeRel = vDirUV - vProbeSize*0.5;
                float vProbeThetai = max(abs(vProbeRel.x), abs(vProbeRel.y));
                float vProbeTheta = acos(dot(vDir, vN));
                info.xyz *= (cos(vProbeTheta - 0.5*3.141592653/vProbeSize) -
                             cos(vProbeTheta + 0.5*3.141592653/vProbeSize))/(4. + 8.*floor(vProbeThetai));
            }
            info.xyz *= dot(vDir, vN);
            
            //Merging
            vec3 lSize = vSize*0.5;
            vec3 lISize = vISize*2.;
            float lProbeSize = vProbeSize*2.;
            if (vLod < 3.5*0. + 3.5) {
                float lPow4 = pow(4., vLod + 1.);
                vec3 lPos = clamp(vPos*vILodFactor*0.5, vec3(0.5), lSize - 0.5);
                vec2 lFaceUV = vec2(0., lOffset + 9.*lSize.z*floor(vFace));
                float lDirIndex0 = floor(vDirUV.x)*2. + floor(vDirUV.y)*lProbeSize*2. + 0.5;
                vec4 lLight = TrilinearCS(lPos, lFaceUV, lDirIndex0, lProbeSize, lSize, lPow4, vPos, vLodFactor*2., lOffset);
                float distInterp = clamp((info.w - vLodFactor)*vILodFactor*0.5, 0., 1.);
                info.xyz = mix(info.xyz, lLight.xyz, distInterp);
            }
        } else {
            info = vec4(0., 0., 0., -1.);
        }
    } else {
        discard;
    }
    fragColor = info;
}