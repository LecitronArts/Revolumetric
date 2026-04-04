/*
Features:
    64*40*64 resolution
    Clamped sampling of voxel-mipmaps (avoids ugly artifacts)
    Temporal mipmaps and better storage
*/

vec3 acesFilm(vec3 x) {
    //Aces film curve
    return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14),0.,1.);
}

vec4 textureCube(vec2 UV) {
    //Samples the cubemap
    vec3 D=vec3(UV*I512-1.,1.);
    return texture(iChannel3,D);
}

vec4 SampleMip0(vec3 sp) {
    sp.y=sp.y-0.5; float fy=floor(sp.y);
    vec2 cuv1=vec2(sp.x+floor(fy*0.2)*64.,sp.z+mod(fy,5.)*64.);
    vec2 cuv2=vec2(sp.x+floor((fy+1.)*0.2)*64.,sp.z+mod(fy+1.,5.)*64.);
    return mix(textureCube(cuv1),
               textureCube(cuv2),fract(sp.y));
}

vec4 SampleMip1(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*32.,sp.z+320.);
    return mix(textureCube(cuv1),
               textureCube(cuv1+vec2(32.,0.)),fract(sp.y));
}

vec4 SampleMip2(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*16.,sp.z+320.+32.);
    return mix(textureCube(cuv1),
               textureCube(cuv1+vec2(16.,0.)),fract(sp.y));
}

vec4 SampleMip3(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*8.,sp.z+320.+48.);
    return mix(textureCube(cuv1),
               textureCube(cuv1+vec2(8.,0.)),fract(sp.y));
}

vec4 SampleMip4(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*4.,sp.z+320.+56.);
    return mix(textureCube(cuv1),
               textureCube(cuv1+vec2(4.,0.)),fract(sp.y));
}

vec4 SampleMip5(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*4.,sp.z+320.+60.);
    return mix(textureCube(cuv1),
               textureCube(cuv1+vec2(2.,0.)),fract(sp.y));
}

vec4 VoxelFetch60NC(vec3 p, float Lod) {
    if (Lod<0.5)
        return SampleMip0(clamp(p,vec3(0.5),vec3(63.5,44.5,63.5)));
    else if (Lod<1.5)
        return SampleMip1(clamp(p*0.5,vec3(0.5),vec3(31.5,44.5,31.5)));
    else if (Lod<2.5)
        return SampleMip2(clamp(p*0.25,vec3(0.5),vec3(15.5,44.5,15.5)));
    else if (Lod<3.5)
        return SampleMip3(clamp(p*0.125,vec3(0.5),vec3(7.5,44.5,7.5)));
    else if (Lod<4.5)
        return SampleMip4(clamp(p*I16,vec3(0.5),vec3(3.5,44.5,3.5)));
    else
        return SampleMip5(clamp(p*I32,vec3(0.5),vec3(1.5,44.5,1.5)));
}

vec4 VoxelFetch60(vec3 p, float Lod) {
    if (Lod<0.5)
        return SampleMip0(clamp(p,vec3(0.5),vec3(63.5,44.5,63.5)))
        	*(clamp(2.-Box(p-vec3(0.5),vec3(63.5,44.5,63.5))*2.,0.,1.));
    else if (Lod<1.5)
        return SampleMip1(clamp(p*0.5,vec3(0.5),vec3(31.5,44.5,31.5)))
        	*(clamp(2.-Box(p*0.5-vec3(0.5),vec3(31.5,44.5,31.5))*2.,0.,1.));
    else if (Lod<2.5)
        return SampleMip2(clamp(p*0.25,vec3(0.5),vec3(15.5,44.5,15.5)))
        	*(clamp(2.-Box(p*0.25-vec3(0.5),vec3(15.5,44.5,15.5))*2.,0.,1.));
    else if (Lod<3.5)
        return SampleMip3(clamp(p*0.125,vec3(0.5),vec3(7.5,44.5,7.5)))
        	*(clamp(2.-Box(p*0.125-vec3(0.5),vec3(7.5,44.5,7.5))*2.,0.,1.));
    else if (Lod<4.5)
        return SampleMip4(clamp(p*I16,vec3(0.5),vec3(3.5,44.5,3.5)))
    		*(clamp(2.-Box(p*I16-vec3(0.5),vec3(3.5,44.5,3.5))*2.,0.,1.));
    else
        return SampleMip5(clamp(p*I32,vec3(0.5),vec3(1.5,44.5,1.5)))
    		*(clamp(2.-Box(p*I32-vec3(0.5),vec3(1.5,44.5,1.5))*2.,0.,1.));
}

vec4 VoxelFetch(vec3 p, float Lod) {
    if (Lod<1.) return mix(VoxelFetch60(p,0.),VoxelFetch60(p,1.),Lod);
    else if (Lod<2.) return mix(VoxelFetch60(p,1.),VoxelFetch60(p,2.),Lod-1.);
    else if (Lod<3.) return mix(VoxelFetch60(p,2.),VoxelFetch60(p,3.),Lod-2.);
    else if (Lod<4.) return mix(VoxelFetch60(p,3.),VoxelFetch60(p,4.),Lod-3.);
    else if (Lod<5.) return mix(VoxelFetch60(p,4.),VoxelFetch60(p,5.),Lod-4.);
    else return VoxelFetch60(p,5.);
}

vec4 Cone(vec3 p, vec3 d, float CR, float Start) {
    vec4 Light=vec4(0.);
    vec3 sp; float t=Start;
    float sD,Lod; vec4 sC;
    for (int i=0; i<58; i++) {
        sp=p+d*t;
        sD=max(1.,t*CR); Lod=log2(sD);
        sC=VoxelFetch(sp,Lod);
        Light+=sC*(1.-Light.w);
        t+=sD;
    }
    return Light+vec4(vec3(0.1,0.3,0.5)*((d.y*0.5+0.5)*(1.-Light.w))*0.5,0.);
}

vec4 Cone60(vec3 p, vec3 d) {
    vec4 Light=vec4(0.);
    vec3 sp; float t=1.; float sD=1.; float Lod=0.;
    for (int i=0; i<7; i++) {
        sp=p+d*t;
        Light+=VoxelFetch60(sp,Lod)*(1.-Light.w);
        t+=sD; sD*=2.; Lod+=1.;
    }
    return Light+vec4(vec3(0.1,0.3,0.5)*((d.y*0.5+0.5)*(1.-Light.w))*0.25,0.);
}


//####
//	50 voxlar i höjd för mipmap0
//		mipmap1 har 20*2=40 voxlar i höjd
//####

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    //Camera
    vec2 uv=fragCoord.xy*IRES;
    vec3 SunDir=texture(iChannel0,vec2(5.5,0.5)*IRES).xyz;
    float Rot=-(iMouse.x*IRES.x)*3.14*3.*0.1;
	vec3 Pos=texture(iChannel0,vec2(3.5,0.5)*IRES).xyz;
    mat3 MM=TBN(texture(iChannel0,vec2(2.5,0.5)*IRES).xyz);
    vec3 Dir=normalize(vec3((uv*2.-1.)*(ASPECT*CFOV),1.)*MM);
	//Trace
    vec3 Color=vec3(0.);
    HIT Pixel;
    if (TraceRay(Pos,Dir,Pixel,iTime)) {
        if (Pixel.Mat==2.) {
            //Emissive
            Color=Pixel.C;
        } else if (Pixel.Mat<1.) {
            //Specular
            Color=Cone(Pixel.P*8.+Pixel.N,reflect(Dir,Pixel.N),Pixel.Mat,
                       texture(iChannel2,uv).x).xyz;
            Color=Color*Pixel.C;
        } else {
            //Diffuse
            vec3 HP=Pixel.P*8.+Pixel.N*0.5;
            mat3 MM=TBN(Pixel.N);
            Color+=(Cone60(HP,Pixel.N).xyz
                    +Cone60(HP,vec3(0.707,0.,0.707)*MM).xyz*0.7
                    +Cone60(HP,vec3(-0.707,0.,0.707)*MM).xyz*0.7
                    +Cone60(HP,vec3(0.,0.707,0.707)*MM).xyz*0.7
                    +Cone60(HP,vec3(0.,-0.707,0.707)*MM).xyz*0.7)*0.2;
            Color=Color*Pixel.C;
        }
    } else {
        Color=(1.-Dir.y*0.5)*vec3(0.1,0.3,0.5)*0.5;
    }
    //Exp color space
    Color=acesFilm(Color);
    //Return
    fragColor=vec4(pow(Color,vec3(0.45)),1.);
}
