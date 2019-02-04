#version 300 es

precision highp float;
precision highp int;
precision highp sampler2D;

uniform float uCameraUnderWater;
uniform vec3 uSunDirection;

uniform mat4 uShortBoxInvMatrix;
uniform mat3 uShortBoxNormalMatrix;
uniform mat4 uTallBoxInvMatrix;
uniform mat3 uTallBoxNormalMatrix;

#include <pathtracing_uniforms_and_defines>
#include <pathtracing_calc_fresnel_reflectance>

uniform sampler2D t_PerlinNoise;

#include <pathtracing_skymodel_defines>

uniform sampler2D tTriangleTexture;
uniform sampler2D tAABBTexture;
uniform sampler2D tAlbedoTextures[8]; // 8 = max number of diffuse albedo textures per model

//float InvTextureWidth = 0.000244140625; // (1 / 4096 texture width)
//float InvTextureWidth = 0.00048828125;  // (1 / 2048 texture width)
//float InvTextureWidth = 0.0009765625;   // (1 / 1024 texture width)

#define INV_TEXTURE_WIDTH 0.00048828125

#define N_QUADS 1

//-----------------------------------------------------------------------

struct Ray { vec3 origin; vec3 direction; };
struct Quad { vec3 v0; vec3 v1; vec3 v2; vec3 v3; vec3 emission; vec3 color; int type; };
struct Sphere { float radius; vec3 position; vec3 emission; vec3 color; int type; };
struct Box { vec3 minCorner; vec3 maxCorner; vec3 emission; vec3 color; int type; };
struct Intersection { vec3 normal; vec3 emission; vec3 color; vec2 uv; int type; int albedoTextureID; float opacity;}; // HACK remove albedoTextureID

Quad quads[N_QUADS];

#include <pathtracing_random_functions>
#include <pathtracing_sphere_intersect>
#include <pathtracing_plane_intersect>
#include <pathtracing_triangle_intersect>
#include <pathtracing_box_intersect>
#include <pathtracing_physical_sky_functions>
#include <pathtracing_boundingbox_intersect>
#include <pathtracing_bvhTriangle_intersect>

//----------------------------------------------------------------------------
float QuadIntersect( vec3 v0, vec3 v1, vec3 v2, vec3 v3, Ray r )
//----------------------------------------------------------------------------
{
	float tTri1 = TriangleIntersect( v0, v1, v2, r );
	float tTri2 = TriangleIntersect( v0, v2, v3, r );
	return min(tTri1, tTri2);
}

//---------------------------------------------------------------------------------------------------------
float DisplacementBoxIntersect( vec3 minCorner, vec3 maxCorner, Ray r )
//---------------------------------------------------------------------------------------------------------
{
	vec3 invDir = 1.0 / r.direction;
	vec3 tmin = (minCorner - r.origin) * invDir;
	vec3 tmax = (maxCorner - r.origin) * invDir;
	
	vec3 real_min = min(tmin, tmax);
	vec3 real_max = max(tmin, tmax);
	
	float minmax = min( min(real_max.x, real_max.y), real_max.z);
	float maxmin = max( max(real_min.x, real_min.y), real_min.z);
	
	// early out
	if (minmax < maxmin) return INFINITY;
	
	if (maxmin > 0.0) // if we are outside the box
	{
		return maxmin;	
	}
		
	if (minmax > 0.0) // else if we are inside the box
	{
		return minmax;
	}
				
	return INFINITY;
}


// SEA
/* Credit: some of the following ocean code is borrowed from https://www.shadertoy.com/view/Ms2SD1 posted by user 'TDM' */

#define SEA_HEIGHT     1.0 // this is how many units from the top of the ocean bounding box
#define SEA_FREQ       1.5 // wave density: lower = spread out, higher = close together
#define SEA_CHOPPY     2.0 // smaller beachfront-type waves, they travel in parallel
#define SEA_SPEED      0.15 // how quickly time passes
#define OCTAVE_M   mat2(1.6, 1.2, -1.2, 1.6);

float hash( vec2 p )
{
	float h = dot(p,vec2(127.1,311.7));	
    	return fract(sin(h)*43758.5453123);
}

float noise( in vec2 p )
{
	vec2 i = floor( p );
	vec2 f = fract( p );	
	vec2 u = f*f*(3.0-2.0*f);
	return -1.0+2.0*mix( mix( hash( i + vec2(0.0,0.0) ), 
		     hash( i + vec2(1.0,0.0) ), u.x),
		mix( hash( i + vec2(0.0,1.0) ), 
		     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

float sea_octave( vec2 uv, float choppy )
{
	uv += noise(uv);        
	vec2 wv = 1.0 - abs(sin(uv));
	vec2 swv = abs(cos(uv));    
	wv = mix(wv, swv, wv);
	return pow(1.0 - pow(wv.x * wv.y, 0.65), choppy);
}

float getOceanWaterHeight( vec3 p )
{
	p.x *= 0.001;
	p.z *= 0.001;
	float freq = SEA_FREQ;
	float amp = SEA_HEIGHT;
	float choppy = SEA_CHOPPY;
	float sea_time = uTime * SEA_SPEED;
	
	vec2 uv = p.xz; uv.x *= 0.75;
	float d, h = 0.0;

	d =  sea_octave((uv + sea_time) * freq, choppy);
	d += sea_octave((uv - sea_time) * freq, choppy);
	h += d * amp;        
	
	return 50.0 * h - 10.0;
}

float getOceanWaterHeight_Detail( vec3 p )
{
	p.x *= 0.001;
	p.z *= 0.001;
	float freq = SEA_FREQ;
	float amp = SEA_HEIGHT;
	float choppy = SEA_CHOPPY;
	float sea_time = uTime * SEA_SPEED;
	
	vec2 uv = p.xz; uv.x *= 0.75;
	float d, h = 0.0;    
	for(int i = 0; i < 4; i++)
	{        
		d =  sea_octave((uv + sea_time) * freq, choppy);
		d += sea_octave((uv - sea_time) * freq, choppy);
		h += d * amp;        
		uv *= OCTAVE_M; freq *= 1.9; amp *= 0.22;
		choppy = mix(choppy, 1.0, 0.2);
	}
	return 50.0 * h - 10.0;
}


// CLOUDS
/* Credit: some of the following cloud code is borrowed from https://www.shadertoy.com/view/XtBXDw posted by user 'valentingalea' */

#define THICKNESS      25.0
#define ABSORPTION     0.45
#define N_MARCH_STEPS  12
#define N_LIGHT_STEPS  3

float noise3D( in vec3 p )
{
	return texture2D(t_PerlinNoise, p.xz).x;
}

const mat3 m = 1.21 * mat3( 0.00,  0.80,  0.60,
                    -0.80,  0.36, -0.48,
		    -0.60, -0.48,  0.64 );

float fbm( vec3 p )
{
	float t;
	float mult = 2.0;
	t  = 1.0 * noise3D(p);   p = m * p * mult;
	t += 0.5 * noise3D(p);   p = m * p * mult;
	t += 0.25 * noise3D(p);
	
	return t;
}

float cloud_density( vec3 pos, float cov )
{
	float dens = fbm(pos * 0.002);
	dens *= smoothstep(cov, cov + 0.05, dens);

	return clamp(dens, 0.0, 1.0);	
}

float cloud_light( vec3 pos, vec3 dir_step, float cov )
{
	float T = 1.0; // transmitance
    	float dens;
    	float T_i;
	
	for (int i = 0; i < N_LIGHT_STEPS; i++) 
	{
		dens = cloud_density(pos, cov);
		T_i = exp(-ABSORPTION * dens);
		T *= T_i;
		pos += dir_step;
	}

	return T;
}

vec4 render_clouds( Ray eye, vec3 p, vec3 sunDirection )
{
	float march_step = THICKNESS / float(N_MARCH_STEPS);
	vec3 pos = p + vec3(uTime * -3.0, uTime * -0.5, uTime * -2.0);
	vec3 dir_step = eye.direction / clamp(eye.direction.y, 0.3, 1.0) * march_step;
	vec3 light_step = sunDirection * 5.0;
	
	float covAmount = (sin(mod(uTime * 0.1, TWO_PI))) * 0.5 + 0.5;
	float coverage = mix(1.0, 1.5, clamp(covAmount, 0.0, 1.0));
	float T = 1.0; // transmitance
	vec3 C = vec3(0); // color
	float alpha = 0.0;
	float dens;
	float T_i;
	float cloudLight;
	
	for (int i = 0; i < N_MARCH_STEPS; i++)
	{
		dens = cloud_density(pos, coverage);

		T_i = exp(-ABSORPTION * dens * march_step);
		T *= T_i;
		cloudLight = cloud_light(pos, light_step, coverage);
		C += T * cloudLight * dens * march_step;
		C = mix(C * 0.95, C, cloudLight);
		alpha += (1.0 - T_i) * (1.0 - alpha);
		pos += dir_step;
	}
	
	return vec4(C, alpha);
}

float checkCloudCover( vec3 sunDirection, vec3 p )
{
	float march_step = THICKNESS / float(N_MARCH_STEPS);
	vec3 pos = p + vec3(uTime * -3.0, uTime * -0.5, uTime * -2.0);
	vec3 dir_step = sunDirection / clamp(sunDirection.y, 0.001, 1.0) * march_step;
	
	float covAmount = (sin(mod(uTime * 0.1, TWO_PI))) * 0.5 + 0.5;
	float coverage = mix(1.0, 1.5, clamp(covAmount, 0.0, 1.0));
	float alpha = 0.0;
	float dens;
	float T_i;
	
	for (int i = 0; i < N_MARCH_STEPS; i++)
	{
		dens = cloud_density(pos, coverage);
		T_i = exp(-ABSORPTION * dens * march_step);
		alpha += (1.0 - T_i) * (1.0 - alpha);
		pos += dir_step;
	}
	
	return clamp(1.0 - alpha, 0.0, 1.0);
}

struct StackLevelData
{
        float id;
        float rayT;
} stackLevels[24];

struct BoxNode
{
	float branch_A_Index;
	vec3 minCorner;
	float branch_B_Index;
	vec3 maxCorner;
};

BoxNode GetBoxNode(const in float i)
{
	// each bounding box's data is encoded in 2 rgba(or xyzw) texture slots
	float iX2 = (i * 2.0);
	// (iX2 + 0.0) corresponds to .x: idLeftChild, .y: aabbMin.x, .z: aabbMin.y, .w: aabbMin.z
	// (iX2 + 1.0) corresponds to .x: idRightChild .y: aabbMax.x, .z: aabbMax.y, .w: aabbMax.z

	vec2 uv0 = vec2( (mod(iX2 + 0.0, 2048.0)), floor((iX2 + 0.0) * INV_TEXTURE_WIDTH) ) * INV_TEXTURE_WIDTH;
	vec2 uv1 = vec2( (mod(iX2 + 1.0, 2048.0)), floor((iX2 + 1.0) * INV_TEXTURE_WIDTH) ) * INV_TEXTURE_WIDTH;

	vec4 aabbNodeData0 = texture( tAABBTexture, uv0 );
	vec4 aabbNodeData1 = texture( tAABBTexture, uv1 );

	BoxNode BN = BoxNode( aabbNodeData0.x,
			      aabbNodeData0.yzw,
			      aabbNodeData1.x,
			      aabbNodeData1.yzw );

        return BN;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
float SceneIntersect( Ray r, inout Intersection intersec, bool checkOcean ) // HACK disable checkOcean
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
	Ray rObj;
	vec3 hitObjectSpace;
	vec3 hitWorldSpace;
	vec3 normal; //vec3 n;
	float dw, dc;
	float d = INFINITY;
	float t = INFINITY;
	float eps = 0.1;
	float waterWaveHeight;

	// AABB BVH Intersection variables
	vec4 aabbNodeData0, aabbNodeData1, aabbNodeData2;
	vec4 vd0, vd1, vd2, vd3, vd4, vd5, vd6, vd7;
	vec3 aabbMin, aabbMax;
	vec3 inverseDir = 1.0 / r.direction;
	vec3 hitPos, toLightBulb;
	vec2 uv0, uv1, uv2, uv3, uv4, uv5, uv6, uv7;

    float stackptr = 0.0;
	float bc, bd;
	float id = 0.0;
	float tu, tv;
	float triangleID = 0.0;
	float triangleU = 0.0;
	float triangleV = 0.0;
	float triangleW = 0.0;

	bool skip = false;
	bool triangleLookupNeeded = false;

	BoxNode currentBoxNode, nodeA, nodeB, tnp;
	StackLevelData currentStackData, slDataA, slDataB, tmp;
	
	// SEA FLOOR
	d = PlaneIntersect(vec4(0, 1, 0, -1000.0), r);
	if (d < t)
	{
		t = d;
		intersec.normal = vec3(0,1,0);
		intersec.emission = vec3(0);
		intersec.color = vec3(0.0, 0.07, 0.07);
		intersec.type = SEAFLOOR;
	}

	for (int i = 0; i < N_QUADS; i++)
        {
		d = QuadIntersect( quads[i].v0, quads[i].v1, quads[i].v2, quads[i].v3, r );
		if (d < t && d > 0.0)
		{
			t = d;
			intersec.normal = normalize( cross(quads[i].v1 - quads[i].v0, quads[i].v2 - quads[i].v0) );
			intersec.emission = quads[i].emission;
			intersec.color = quads[i].color;
			intersec.type = quads[i].type;
		}
    }
	
    // skip rendering the rest
//    return t; // HACK disable this
	
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// OCEAN 
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/*
	if ( !checkOcean )
	{
		return t;
	}

	vec3 pos = r.origin;
	vec3 dir = r.direction;
	float h = 0.0;
	d = 0.0; // reset d

	for(int i = 0; i < 100; i++)
	{
		h = abs(pos.y - getOceanWaterHeight(pos));
		if (d > 4000.0 || h < 1.0) break;
		d += h;
		pos += dir * h;
	}
	hitWorldSpace = pos;

	if (d > 4000.0)
	{
		d = PlaneIntersect( vec4(0, 1, 0, 0.0), r );
		if ( d >= INFINITY ) return t;
		hitWorldSpace = r.origin + r.direction * d;

		waterWaveHeight = getOceanWaterHeight_Detail(hitWorldSpace);
		d = DisplacementBoxIntersect( vec3(-INFINITY, -INFINITY, -INFINITY), vec3(INFINITY, waterWaveHeight, INFINITY), r);
		hitWorldSpace = r.origin + r.direction * d;
	}

	if (d < t)
	{
		float eps = 1.0;
		t = d;
		float dx = getOceanWaterHeight_Detail(hitWorldSpace - vec3(eps,0,0)) - getOceanWaterHeight_Detail(hitWorldSpace + vec3(eps,0,0));
		float dy = eps * 2.0; // (the water wave height is a function of x and z, not dependent on y)
		float dz = getOceanWaterHeight_Detail(hitWorldSpace - vec3(0,0,eps)) - getOceanWaterHeight_Detail(hitWorldSpace + vec3(0,0,eps));

		intersec.normal = normalize(vec3(dx,dy,dz));
		intersec.emission = vec3(0);
		intersec.color = vec3(0.6, 1.0, 1.0);
		intersec.type = REFR;
	}
	*/
	
//	return t; // HACK disable  this
	
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// glTF
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	currentBoxNode = GetBoxNode(stackptr);
	currentStackData = StackLevelData(stackptr, BoundingBoxIntersect(currentBoxNode.minCorner, currentBoxNode.maxCorner, r.origin, inverseDir));
	stackLevels[0] = currentStackData;

	while (true)
        {

		if (currentStackData.rayT < t + 65.0) // 65.0 is the magic number for this scene
                {

                        if (currentBoxNode.branch_A_Index >= 0.0) // signifies this is a branch
                        {
                                nodeA = GetBoxNode(currentBoxNode.branch_A_Index);
                                nodeB = GetBoxNode(currentBoxNode.branch_B_Index);
                                slDataA = StackLevelData(currentBoxNode.branch_A_Index, BoundingBoxIntersect(nodeA.minCorner, nodeA.maxCorner, r.origin, inverseDir));
                                slDataB = StackLevelData(currentBoxNode.branch_B_Index, BoundingBoxIntersect(nodeB.minCorner, nodeB.maxCorner, r.origin, inverseDir));

				// first sort the branch node data so that 'a' is the smallest
				if (slDataB.rayT < slDataA.rayT)
				{
					tmp = slDataB;
					slDataB = slDataA;
					slDataA = tmp;

					tnp = nodeB;
					nodeB = nodeA;
					nodeA = tnp;
				} // branch 'b' now has the larger rayT value of 'a' and 'b'

				if (slDataB.rayT < INFINITY) // see if branch 'b' (the larger rayT) needs to be processed
				{
					currentStackData = slDataB;
					currentBoxNode = nodeB;
					skip = true; // this will prevent the stackptr from decreasing by 1
				}
				if (slDataA.rayT < INFINITY) // see if branch 'a' (the smaller rayT) needs to be processed
				{
					if (skip == true) // if larger branch 'b' needed to be processed also,
						stackLevels[int(stackptr++)] = slDataB; // cue larger branch 'b' for future round
								// also, increase pointer by 1

					currentStackData = slDataA;
					currentBoxNode = nodeA;
					skip = true; // this will prevent the stackptr from decreasing by 1
				}
                        }

                        else //if (currentBoxNode.branch_A_Index < 0.0) //  < 0.0 signifies a leaf node
                        {
				// each triangle's data is encoded in 8 rgba(or xyzw) texture slots
				id = 8.0 * (-currentBoxNode.branch_A_Index - 1.0);
				uv0 = vec2( (mod(id + 0.0, 2048.0)), floor((id + 0.0) * INV_TEXTURE_WIDTH) ) * INV_TEXTURE_WIDTH;
				uv1 = vec2( (mod(id + 1.0, 2048.0)), floor((id + 1.0) * INV_TEXTURE_WIDTH) ) * INV_TEXTURE_WIDTH;
				uv2 = vec2( (mod(id + 2.0, 2048.0)), floor((id + 2.0) * INV_TEXTURE_WIDTH) ) * INV_TEXTURE_WIDTH;

				vd0 = texture( tTriangleTexture, uv0 );
				vd1 = texture( tTriangleTexture, uv1 );
				vd2 = texture( tTriangleTexture, uv2 );

				d = BVH_TriangleIntersect( vec3(vd0.xyz), vec3(vd0.w, vd1.xy), vec3(vd1.zw, vd2.x), r, tu, tv );

				if (d < t && d > 0.0)
				{
					t = d;
					triangleID = id;
					triangleU = tu;
					triangleV = tv;
					triangleLookupNeeded = true;
				}
                        }
		} // end if (currentStackData.rayT < t)

		if (skip == false)
                {
                        // decrease pointer by 1 (0.0 is root level, 24.0 is maximum depth)
                        if (--stackptr < 0.0) // went past the root level, terminate loop
                                break;
                        currentStackData = stackLevels[int(stackptr)];
                        currentBoxNode = GetBoxNode(currentStackData.id);
                }
		skip = false; // reset skip

        } // end while (true)


	if (triangleLookupNeeded)
	{
		uv0 = vec2( (mod(triangleID + 0.0, 2048.0)), floor((triangleID + 0.0) * INV_TEXTURE_WIDTH) ) * INV_TEXTURE_WIDTH;
		uv1 = vec2( (mod(triangleID + 1.0, 2048.0)), floor((triangleID + 1.0) * INV_TEXTURE_WIDTH) ) * INV_TEXTURE_WIDTH;
		uv2 = vec2( (mod(triangleID + 2.0, 2048.0)), floor((triangleID + 2.0) * INV_TEXTURE_WIDTH) ) * INV_TEXTURE_WIDTH;
		uv3 = vec2( (mod(triangleID + 3.0, 2048.0)), floor((triangleID + 3.0) * INV_TEXTURE_WIDTH) ) * INV_TEXTURE_WIDTH;
		uv4 = vec2( (mod(triangleID + 4.0, 2048.0)), floor((triangleID + 4.0) * INV_TEXTURE_WIDTH) ) * INV_TEXTURE_WIDTH;
		uv5 = vec2( (mod(triangleID + 5.0, 2048.0)), floor((triangleID + 5.0) * INV_TEXTURE_WIDTH) ) * INV_TEXTURE_WIDTH;
		uv6 = vec2( (mod(triangleID + 6.0, 2048.0)), floor((triangleID + 6.0) * INV_TEXTURE_WIDTH) ) * INV_TEXTURE_WIDTH;
		uv7 = vec2( (mod(triangleID + 7.0, 2048.0)), floor((triangleID + 7.0) * INV_TEXTURE_WIDTH) ) * INV_TEXTURE_WIDTH;

		vd0 = texture( tTriangleTexture, uv0 );
		vd1 = texture( tTriangleTexture, uv1 );
		vd2 = texture( tTriangleTexture, uv2 );
		vd3 = texture( tTriangleTexture, uv3 );
		vd4 = texture( tTriangleTexture, uv4 );
		vd5 = texture( tTriangleTexture, uv5 );
		vd6 = texture( tTriangleTexture, uv6 );
		vd7 = texture( tTriangleTexture, uv7 );

		// face normal for flat-shaded polygon look
		//intersec.normal = normalize( cross(vec3(vd0.w, vd1.xy) - vec3(vd0.xyz), vec3(vd1.zw, vd2.x) - vec3(vd0.xyz)) );

		// interpolated normal using triangle intersection's uv's
		triangleW = 1.0 - triangleU - triangleV;
		intersec.normal = normalize(triangleW * vec3(vd2.yzw) + triangleU * vec3(vd3.xyz) + triangleV * vec3(vd3.w, vd4.xy));
		intersec.emission = vec3(1, 0, 1); // use this if intersec.type will be LIGHT
		intersec.color = vd6.yzw;
		intersec.opacity = vd7.y;
		intersec.uv = triangleW * vec2(vd4.zw) + triangleU * vec2(vd5.xy) + triangleV * vec2(vd5.zw);
		intersec.type = int(vd6.x);
		intersec.albedoTextureID = int(vd7.x);
	}

	return t;

} // end float SceneIntersect( Ray r, inout Intersection intersec )



//-----------------------------------------------------------------------
vec3 CalculateRadiance( Ray r, vec3 sunDirection, inout uvec2 seed ) // HACK disable sunDirection
//-----------------------------------------------------------------------
{
	vec3 randVec = vec3(rand(seed) * 2.0 - 1.0, rand(seed) * 2.0 - 1.0, rand(seed) * 2.0 - 1.0);
	Ray cameraRay = r;
	vec3 initialSkyColor = Get_Sky_Color(r, sunDirection);
	
	Ray skyRay = Ray( r.origin * vec3(0.02), normalize(vec3(r.direction.x, abs(r.direction.y), r.direction.z)) );
	float dc = SphereIntersect( 20000.0, vec3(skyRay.origin.x, -19900.0, skyRay.origin.z) + vec3(rand(seed) * 2.0), skyRay );
	vec3 skyPos = skyRay.origin + skyRay.direction * dc;
	vec4 cld = render_clouds(skyRay, skyPos, sunDirection);
	
	Ray cloudShadowRay = Ray(r.origin * vec3(0.02), normalize(sunDirection + (randVec * 0.05)));
	float dcs = SphereIntersect( 20000.0, vec3(skyRay.origin.x, -19900.0, skyRay.origin.z) + vec3(rand(seed) * 2.0), cloudShadowRay );
	vec3 cloudShadowPos = cloudShadowRay.origin + cloudShadowRay.direction * dcs;
	float cloudShadowFactor = checkCloudCover(cloudShadowRay.direction, cloudShadowPos);
	
	Intersection intersec;
	vec3 accumCol = vec3(0.0);
    vec3 mask = vec3(1.0);
	vec3 n, nl, x;
	vec3 firstX = vec3(0);
	vec3 checkCol0 = vec3(1);
	vec3 checkCol1 = vec3(0.5);
    vec3 tdir;

	float nc, nt, Re;
	float weight;
	float t = INFINITY;
	
	int previousIntersecType = -1;
	int diffuseCount = 0;
	bool checkOcean = false;
	bool skyHit = false;
    float epsIntersect = 0.01;

	bool bounceIsSpecular = true;


    for (int bounces = 0; bounces < 6; bounces++)
	{

		float t = SceneIntersect(r, intersec, false); // HACK remove false (checkOcean)


		// ray hits sky first
		if (t == INFINITY && bounces == 0 ) // HACK remove bounces
		{
			skyHit = true; // HACK remove
			firstX = skyPos; // HACK remove
			accumCol = initialSkyColor; // HACK remove
			break;	
		}

		// if ray bounced off of water and hits sky
		if ( t == INFINITY && previousIntersecType == REFR )
		{
			if (bounceIsSpecular) // prevents sun 'fireflies' on diffuse surfaces
//				accumCol = mask * Get_Sky_Color(r, sunDirection);
				accumCol = mask * Get_Sky_Color(r, sunDirection) * 1.0 / SUN_INTENSITY;

			//if (uCameraUnderWater > 0.0) // uncomment 'if' for clouds reflection in water, but it's a
//				skyHit = true;       // straight mirror-reflection, which is not physically possible
			 
			
			firstX = skyPos;
			
			break;	
		}
		
		/*
		// if ray bounced off of mirror box and hits sky
		if (t == INFINITY && previousIntersecType == SPEC)
		{
			if (bounceIsSpecular) // prevents sun 'fireflies' on diffuse surfaces
				accumCol = mask * Get_Sky_Color(r, sunDirection);
			
			if (bounces == 1) // reflection of sky in tall mirror box
			{
				initialSkyColor = Get_Sky_Color(Ray(r.origin, vec3(r.direction.x, abs(r.direction.y), r.direction.z)), sunDirection);
				skyRay = Ray(r.origin * 0.01, normalize(vec3(r.direction.x, abs(r.direction.y), r.direction.z)) );
				dc = PlaneIntersect( vec4(0, -1, 0, -150.0 + (rand(seed) * 2.0)), skyRay );
				skyPos = skyRay.origin + skyRay.direction * dc;
				
				skyHit = true;
				firstX = skyPos;
			}
			
			break;	
		}
		*/
		
		// if ray bounced off of diffuse material (short box or walls/floor) and hits sky
		if (t == INFINITY && previousIntersecType == DIFF)
		{	
			weight = 2.0;
			// prevents sun 'fireflies' on diffuse surfaces
			if (bounceIsSpecular || dot(r.direction, sunDirection) > 0.98)
				weight = 0.03;
			 
			accumCol = mask * Get_Sky_Color(r, sunDirection) * weight;
					
			break;
		}

/*
		// HACK might not need this (gltf Loading)
		// if we reached something bright, don't spawn any more rays
		if (intersec.type == LIGHT)
		{
			//if (bounceIsSpecular)
			{
				accumCol = mask * intersec.emission;
			}

			break;
		}
		*/


		// useful data
		vec3 n = intersec.normal;
        vec3 nl = dot(n,r.direction) <= 0.0 ? normalize(n) : normalize(n * -1.0);
		vec3 x = r.origin + r.direction * t;

		if (bounces == 0) 
			firstX = x;

		if (intersec.type == SEAFLOOR)
		{
			float waterDotSun = max(0.0, dot(vec3(0,1,0), sunDirection));
			float waterDotCamera = max(0.4, dot(vec3(0,1,0), -cameraRay.direction));
			accumCol = mask * intersec.color * waterDotSun * waterDotCamera;
                        break;
		}
		
        if (intersec.type == DIFF || intersec.type == CHECK) // Ideal DIFFUSE reflection
        {
			diffuseCount++; // HACK remove this

			previousIntersecType = DIFF; // HACK remove this

			checkOcean = false; // HACK remove this
			
			// HACK might not need this (glTF viewer)
			if( intersec.type == CHECK )
			{
				float q = clamp( mod( dot( floor(x.xz * 0.04), vec2(1.0) ), 2.0 ) , 0.0, 1.0 );
				intersec.color = checkCol0 * q + checkCol1 * (1.0 - q);
			}

			mask *= intersec.color;
//			bounceIsSpecular = false; // HACK enable this, disable below

			/* HACK enable this
			// Russian Roulette
			float p = max(mask.r, max(mask.g, mask.b));
			if (bounces > 0)
			{
				if (rand(seed) < p)
                                	mask *= 1.0 / p;
                        	else
                                	break;
			}
			*/

            if (diffuseCount == 1 && rand(seed) < 0.5) // HACK disable this
            {
				// choose random Diffuse sample vector
				r = Ray( x, randomCosWeightedDirectionInHemisphere(nl, seed) );
				r.origin += r.direction * epsIntersect;
				bounceIsSpecular = false;
				continue;
            }
            else
            {
            	r = Ray( x, normalize(sunDirection + (randVec * 0.01)) );
				r.origin += nl;
				weight = max(0.0, dot(r.direction, nl));
				mask *= clamp(weight, 0.0, 1.0);
				
				bounceIsSpecular = true;
				continue;
            }
        }
/*
        if (intersec.type == SPEC)  // Ideal SPECULAR reflection
        {
			//checkOcean = true;
			mask *= intersec.color; // HACK disable this, disable below
			r = Ray( x, reflect(r.direction, nl) );
			r.origin += r.direction * epsIntersect;
			mask *= intersec.color;
			//bounceIsSpecular = true; // HACK enable this
			previousIntersecType = SPEC;
			
			continue;
        }
        */

        if (intersec.type == REFR)  // Ideal dielectric REFRACTION
		{
			previousIntersecType = REFR;
			checkOcean = false;
			
			nc = 1.0; // IOR of Air
			nt = 1.5; // IOR of common Glass
			Re = calcFresnelReflectance(n, nl, r.direction, nc, nt, tdir);

			if (diffuseCount < 2)
				bounceIsSpecular = true; // HACK enable this

			if (rand(seed) < Re) // reflect ray from surface
			{
				r = Ray( x, reflect(r.direction, nl) );
				r.origin += r.direction * epsIntersect;
			    	continue;
			}
			else // transmit ray through surface
			{
				mask *= intersec.color * (1.0 - intersec.opacity);
				r = Ray(x, tdir);
				r.origin += r.direction * epsIntersect;
				continue;
			}

		} // end if (intersec.type == REFR)

/*
		// HACK disable wood
		if (intersec.type == WOOD)  // Diffuse object underneath with thin layer of Water on top
		{
			checkOcean = false;

			float roughness = 0.2;

			nc = 1.0; // IOR of air
			nt = 1.1; // IOR of ClearCoat
			Re = calcFresnelReflectance(n, nl, r.direction, nc, nt, tdir);

			// choose either specular reflection or diffuse
			if ( rand(seed) < Re )
			{
				vec3 reflectVec = reflect(r.direction, nl);
				r = Ray( x, mix( reflectVec, normalize(nl + randVec), roughness) );
				r.origin += r.direction;
				previousIntersecType = REFR;

				bounceIsSpecular = (diffuseCount < 2);

				continue;
			}
			else
			{
				previousIntersecType = DIFF;

				float pattern = abs(noise(vec2( (x.x * 0.5 * x.z * 0.5 + sin(x.y*0.005)) )));
				float woodPattern = 1.0 / max(1.0, pattern * 100.0);
				intersec.color *= vec3(woodPattern);

				mask *= intersec.color;

				if (rand(seed) < 0.5)
				{
					// choose random Diffuse sample vector
					r = Ray( x, randomCosWeightedDirectionInHemisphere(nl, seed) );
					r.origin += r.direction;

					bounceIsSpecular = false;
					continue;
				}
				else
				{
					r = Ray( x, normalize(sunDirection + (randVec * 0.01)) );
					r.origin += nl;
					weight = max(0.0, dot(r.direction, nl));
					mask *= clamp(weight, 0.0, 1.0);

					bounceIsSpecular = true;
					continue;
				}
			}

		} //end if (intersec.type == WOOD)
		
		if (intersec.type == COAT)  // Diffuse object underneath with ClearCoat on top (like car, or shiny pool ball)
		{
			nc = 1.0; // IOR of Air
			nt = 1.4; // IOR of ClearCoat
			Re = calcFresnelReflectance(n, nl, r.direction, nc, nt, tdir);

			// choose either specular reflection or diffuse
			if( rand(seed) < Re )
			{
				r = Ray( x, reflect(r.direction, nl) );
				r.origin += r.direction * epsIntersect;
				bounceIsSpecular = true;
				continue;
			}
			else
			{
				mask *= intersec.color;

				int id = intersec.albedoTextureID;
				if (id > -1)
				{
					vec3 albedoSample;
					     if (id == 0) albedoSample = texture(tAlbedoTextures[0], intersec.uv).rgb;
					else if (id == 1) albedoSample = texture(tAlbedoTextures[1], intersec.uv).rgb;
					else if (id == 2) albedoSample = texture(tAlbedoTextures[2], intersec.uv).rgb;
					else if (id == 3) albedoSample = texture(tAlbedoTextures[3], intersec.uv).rgb;
					else if (id == 4) albedoSample = texture(tAlbedoTextures[4], intersec.uv).rgb;
					else if (id == 5) albedoSample = texture(tAlbedoTextures[5], intersec.uv).rgb;
					else if (id == 6) albedoSample = texture(tAlbedoTextures[6], intersec.uv).rgb;
					else if (id == 7) albedoSample = texture(tAlbedoTextures[7], intersec.uv).rgb;

					mask *= albedoSample;
				}

				r = Ray( x, randomCosWeightedDirectionInHemisphere(nl, seed) );
				r.origin += r.direction * epsIntersect;
				bounceIsSpecular = false;
				continue;
			}

		} //end if (intersec.type == COAT)
		*/


	} // end for (int bounces = 0; bounces < 5; bounces++)

	// atmospheric haze effect (aerial perspective)
	float hitDistance;
	
	if ( skyHit ) // sky and clouds
	{
		vec3 cloudColor = cld.rgb / (cld.a + 0.00001);
		vec3 sunColor = clamp(Get_Sky_Color( Ray(skyPos, normalize((randVec * 0.03) + sunDirection)), sunDirection ), 0.0, 1.0);
		
		cloudColor *= mix(sunColor, vec3(1), max(0.0, dot(vec3(0,1,0), sunDirection)) );
		cloudColor = mix(initialSkyColor, cloudColor, clamp(cld.a, 0.0, 1.0));
		
		hitDistance = distance(skyRay.origin, skyPos);
		accumCol = mask * mix( accumCol, cloudColor, clamp( exp2( -hitDistance * 0.004 ), 0.0, 1.0 ) );
	}	
	else // terrain and other objects
	{
		hitDistance = distance(cameraRay.origin, firstX);
		accumCol = mix( initialSkyColor, accumCol, clamp( exp2( -log(hitDistance * 0.00003) ), 0.0, 1.0 ) );

		// underwater fog effect
		//hitDistance = distance(cameraRay.origin, firstX);
		//hitDistance *= 0.0;
		//accumCol = mix( vec3(0.0,0.05,0.05), accumCol, clamp( exp2( -hitDistance * 0.001 ), 0.0, 1.0 ) );
	}

	return vec3(max(vec3(0), accumCol));   // HACK disable this, enable accumCol below

	//return accumCol;
}


//-----------------------------------------------------------------------
void SetupScene(void)
//-----------------------------------------------------------------------
{
	float x = INFINITY;
	quads[0] = Quad( vec3( x/-2.0, 0, x/2.0), vec3(x/2.0, 0, x/2.0), vec3(x/2.0, 0, x/-2.0), vec3(x/-2.0, 0, x/-2.0),    vec3(0), vec3(0.45), DIFF);// Floor
}


//#include <pathtracing_main> // HACK enable this, disable below

// tentFilter from Peter Shirley's 'Realistic Ray Tracing (2nd Edition)' book, pg. 60		
float tentFilter(float x)
{
	if (x < 0.5) 
		return sqrt(2.0 * x) - 1.0;
	else return 1.0 - sqrt(2.0 - (2.0 * x));
}

// cubicSplineFilter from Peter Shirley's 'Realistic Ray Tracing (2nd Edition)' book, pg. 58
float solve(float r)
{
	float u = r;
	for (int i = 0; i < 5; i++)
	{
		u = (11.0 * r + u * u * (6.0 + u * (8.0 - 9.0 * u))) /
			(4.0 + 12.0 * u * (1.0 + u * (1.0 - u)));
	}
	return u;
}

float cubicFilter(float x)
{
	if (x < 1.0 / 24.0)
		return pow(24.0 * x, 0.25) - 2.0;
	else if (x < 0.5)
		return solve(24.0 * (x - 1.0 / 24.0) / 11.0) - 1.0;
	else if (x < 23.0 / 24.0)
		return 1.0 - solve(24.0 * (23.0 / 24.0 - x) / 11.0);
	else return 2.0 - pow(24.0 * (1.0 - x), 0.25);
}
void main( void )
{
	// not needed, three.js has a built-in uniform named cameraPosition
	//vec3 camPos   = vec3( uCameraMatrix[3][0],  uCameraMatrix[3][1],  uCameraMatrix[3][2]);
	
    	vec3 camRight   = vec3( uCameraMatrix[0][0],  uCameraMatrix[0][1],  uCameraMatrix[0][2]);
    	vec3 camUp      = vec3( uCameraMatrix[1][0],  uCameraMatrix[1][1],  uCameraMatrix[1][2]);
	vec3 camForward = vec3(-uCameraMatrix[2][0], -uCameraMatrix[2][1], -uCameraMatrix[2][2]);
	
	// seed for rand(seed) function
	uvec2 seed = uvec2(uFrameCounter, uFrameCounter + 1.0) * uvec2(gl_FragCoord);

	vec2 pixelPos = vec2(0);
	vec2 pixelOffset = vec2(0);
	
	float x = rand(seed);
	float y = rand(seed);

	if (!uCameraIsMoving)
	{
		pixelOffset.x = tentFilter(x);
		pixelOffset.y = tentFilter(y);
	}
	
	// pixelOffset ranges from -1.0 to +1.0, so only need to divide by half resolution
	pixelOffset /= (uResolution * 0.5);

	// vUv comes in the range 0.0 to 1.0, so we must map it to the range -1.0 to +1.0
	pixelPos = vUv * 2.0 - 1.0;
	pixelPos += pixelOffset;

	vec3 rayDir = normalize( pixelPos.x * camRight * uULen + pixelPos.y * camUp * uVLen + camForward );
	
	// depth of field
	vec3 focalPoint = uFocusDistance * rayDir;
	float randomAngle = rand(seed) * TWO_PI; // pick random point on aperture
	float randomRadius = rand(seed) * uApertureSize;
	vec3  randomAperturePos = ( cos(randomAngle) * camRight + sin(randomAngle) * camUp ) * randomRadius;
	// point on aperture to focal point
	vec3 finalRayDir = normalize(focalPoint - randomAperturePos);
    
	Ray ray = Ray( cameraPosition + randomAperturePos, finalRayDir );

	SetupScene(); 

	// perform path tracing and get resulting pixel color
	vec3 pixelColor = CalculateRadiance( ray, uSunDirection, seed ); // HACK disable uSunDirection
	
	vec3 previousColor = texture2D(tPreviousTexture, vUv).rgb; // HACK use texture
	
	if ( uCameraJustStartedMoving )
	{
		previousColor = vec3(0.0); // clear rendering accumulation buffer
	}
	else if ( uCameraIsMoving )
	{
		previousColor *= 0.5; // motion-blur trail amount (old image) // UPDATE use 0.78 for better performance, use in different state?     else if ( uCameraIsMoving || rayHitIsDynamic )
		pixelColor *= 0.5; // brightness of new image (noisy) // UPDATE use 0.22 for better performance, use in different state?     else if ( uCameraIsMoving || rayHitIsDynamic )
	}
	
	
	out_FragColor = vec4( pixelColor + previousColor, 1.0 );	
}