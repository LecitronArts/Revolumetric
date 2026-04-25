//Voxels

const float YLimit2=32.+16.+8.+4.+2.+1.;
const float YOffset[5]=float[5](0.,32.,48.,56.,60.);
const float FetchSize[5]=float[5](16.,8.,4.,2.,1.);

bool SS(vec3 sp) { //Sample Scene
    return SDF(sp*(I64*8.),iTime).D<I16;
}

float VTrace(vec3 p, vec3 d) {
    vec3 sp;
    for (float i=0.; i<46.; i++) {
        sp=p+d*i;
        if (sp.x<0.5 || sp.x>63.5 || sp.z<0.5 || sp.z>63.5 || sp.y<0.5 || sp.y>31.5) break;
        if (SS(sp)) return 0.;
    }
    return 1.;
}

void mainCubemap(out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir) {
    vec4 Output=vec4(0.);
    vec2 UV; vec3 aDir=abs(rayDir);
    if (aDir.z>max(aDir.x,aDir.y)) {
        //Z-side
        UV = floor(((rayDir.xy/aDir.z)*0.5+0.5)*1024.)+0.5;
        if (rayDir.z>0.) {
            //Positive z-side
            if (UV.y<320.) {
                //LOD 0
                vec2 fragFloor=floor(UV);
                vec3 Pos=vec3(mod(fragFloor.x,64.),
                            floor(fragFloor.y*I64)+floor(fragFloor.x*I64)*5.,
                            mod(fragFloor.y,64.))+0.5;
                DF Sample=SDF(Pos*(I64*8.),iTime);
                if (Sample.D<I16) {
                    //Voxel exists
                    float Weight=min(0.9999,(1.-Sample.D/I16));
                    if (Sample.Mat>1.5) {
                        //Emissive
                        Output.xyz=Sample.C;
                    } else {
                        //Diffuse or glossy
                    }
                    Output=vec4(Output.xyz*Weight,Weight);
                }
            }
            if (UV.y>320. && UV.y<384.) {
                vec2 fragFloor=floor(vec2(UV.x,UV.y-320.));
                if (fragFloor.y<32.) {
                    //MipMap1 samples MipMap0 directly
                    float lody=floor(fragFloor.x*I32)*2.;
                    vec2 Offset1=vec2(floor(lody*0.2)*64.,mod(lody,5.)*64.);
                    vec2 Offset2=vec2(floor((lody+1.)*0.2)*64.,mod(lody+1.,5.)*64.);
                    vec2 fuv=mod(fragFloor,32.);
                    Output=(texture(iChannel3,vec3((fuv*2.+1.+Offset1)*I512-1.,1.))+
                           texture(iChannel3,vec3((fuv*2.+1.+Offset2)*I512-1.,1.)))*0.5;
                } else if (fragFloor.y<YLimit2) {
                    //Temporal, samples this buffer
                    int Index=int(4.-floor(log2(YLimit2-fragFloor.y)));
                    float Size=FetchSize[Index];
                    float ISize=1./Size;
                    vec2 Offset=vec2(floor(fragFloor.x*ISize)*2.*(Size*2.),
                                     320.+YOffset[Index]);
                    vec2 fuv=mod(fragFloor,Size);
                    Output=(texture(iChannel3,vec3((fuv*2.+1.+Offset)*I512-1.,1.))+
                           texture(iChannel3,vec3((fuv*2.+1.+Offset+vec2(Size*2.,0.))*I512-1.,1.)))*0.5;
                }
            }
        }
    }
    fragColor=Output;
}
