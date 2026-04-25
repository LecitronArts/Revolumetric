//Shadow map

vec4 TextureCube(vec2 uv) {
    float tcSign = -mod(floor(uv.y*I1024), 2.)*2. + 1.;
    vec3 tcD = vec3(vec2(uv.x, mod(uv.y, 1024.))*I512 - 1., tcSign);
    if (uv.y > 4096.) tcD = tcD.xzy;
    else if (uv.y > 2048.) tcD = tcD.zxy;
    return textureLod(iChannel3, tcD, 0.);
}

vec4 Trace(vec3 p, vec3 d) {
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
        if (sC.w > 0.5) return vec4(t);
        lfp = fp;
        fp += ABoxfarNormal(p, iDir, signdir, fp, fp + 1., t);
    }
    return vec4(-10.);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec3 SUN_DIR = GetSunDir(iTime);
    vec3 SUN_TAN = normalize(cross(SUN_DIR, vec3(0., 1., 0.)));
    vec3 SUN_BIT = cross(SUN_TAN, SUN_DIR);
    vec2 sun_uv = (fragCoord.xy*IRES*2. - 1.)*ASPECT*SUN_SM_SIZE;
    vec3 SUN_POS = vec3(16., 16., 24.) + SUN_DIR*SUN_DIST + SUN_TAN*sun_uv.x + SUN_BIT*sun_uv.y;
    vec4 info = Trace(SUN_POS, -SUN_DIR);
    fragColor = info;
}