import numpy as np
import trimesh
import math
from scipy.spatial.transform import Rotation as sciR

class RigidMatrix():
    def __init__(self, data):
        assert data.shape == (4,4)
        self._data = data
    
    @classmethod
    def fromRt(cls, R, t):
        M = np.zeros([4,4])
        M[:3, :3] = R
        M[-1, :3] = t
        return RigidMatrix(M)
    
    @property
    def R(self):
        return self._data[:3,:3]
    
    @property
    def t(self):
        return self._data[-1,:3]
    
    @property
    def data(self):
        return self._data
    
    @property
    def T(self):
        return self.inverse()
    
    def __add__(self, other):
        return RigidMatrix(self.data + other.data)
    
    def __mul__(self, other):
        return RigidMatrix(self.data @ other.data)
    
    def inverse(self):
        return RigidMatrix.fromRt(self.R.T, -self.t)

def rotationMatrixToEulerAngles(R):
    # assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def rand_rotation_matrix(deflection=1.0, randnums=None, makeT=False):
    """
    Creates a random rotation matrix.
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
    theta, phi, z = randnums
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    st = np.sin(theta)
    ct = np.cos(theta)
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    if makeT:
        T = np.identity(4)
        T[0, :3] = M[0]
        T[1, :3] = M[1]
        T[2, :3] = M[2]
        return T
    else:
        return M

# functions for so3 sampling
def get_adjmatrix_trimesh(mesh, gsize=None):
    face_idx = mesh.faces
    face_adj = mesh.face_adjacency
    adj_idx = []
    binary_swap = np.vectorize(lambda a: 1 if a == 0 else 0)
    
    for i, fidx in enumerate(face_idx):
        fid = np.argwhere(face_adj == i)
        if len(fid) > 0:
            fid[:,1] = binary_swap(fid[:,1])
            adj_result = face_adj[tuple(np.split(fid, 2, axis=1))].T
            adj_idx.append(adj_result)
        else:
            # Handle case where no adjacency is found
            adj_idx.append(np.array([]))
    
    # Robust handling of array stacking
    if len(adj_idx) > 0:
        # Filter out empty arrays
        non_empty_adj = [arr for arr in adj_idx if len(arr) > 0]
        
        if len(non_empty_adj) > 0:
            # Find the most common shape
            shapes = [arr.shape for arr in non_empty_adj]
            from collections import Counter
            most_common_shape = Counter(shapes).most_common(1)[0][0]
            
            # Keep only arrays with the most common shape
            filtered_adj = [arr for arr in non_empty_adj if arr.shape == most_common_shape]
            
            if len(filtered_adj) > 0:
                try:
                    face_adj = np.vstack(filtered_adj).astype(np.int32)
                except ValueError as e:
                    print(f"Warning: Could not stack face adjacency arrays: {e}")
                    face_adj = np.array([], dtype=np.int32).reshape(0, 2)
            else:
                face_adj = np.array([], dtype=np.int32).reshape(0, 2)
        else:
            face_adj = np.array([], dtype=np.int32).reshape(0, 2)
    else:
        face_adj = np.array([], dtype=np.int32).reshape(0, 2)
    
    if gsize is None:
        return face_adj
    else:
        # Padding with in-plane rotation neighbors
        if face_adj.size > 0:
            na = face_adj.shape[0]
            R_adj = (face_adj * gsize)[:,None].repeat(gsize, axis=1).reshape(-1,3)
            R_adj = np.tile(R_adj,[1,gsize]) + np.arange(gsize).repeat(3)[None].repeat(na*gsize, axis=0)
            rp = (np.arange(na) * gsize).repeat(gsize)[..., None].repeat(gsize,axis=1)
            rp = rp + np.arange(gsize)[None].repeat(na*gsize,axis=0)
            R_adj = np.concatenate([R_adj, rp], axis=1)
            return R_adj
        else:
            # Return empty array with correct shape
            return np.array([], dtype=np.int32).reshape(0, 6)

def get_so3_from_anchors_np(face_normals, gsize=3):
    # alpha, beta
    na = face_normals.shape[0]
    sbeta = face_normals[...,-1]
    cbeta = (1 - sbeta**2)**0.5
    
    # Avoid division by zero
    cbeta = np.where(cbeta < 1e-8, 1e-8, cbeta)
    
    calpha = face_normals[...,0] / cbeta
    salpha = face_normals[...,1] / cbeta
    
    # gamma
    gamma = np.linspace(0, 2 * np.pi, gsize, endpoint=False, dtype=np.float32)
    gamma = -gamma[None].repeat(na, axis=0)
    
    # Compute na rotation matrices Rx, Ry, Rz
    Rz = np.zeros([na, 9], dtype=np.float32)
    Ry = np.zeros([na, 9], dtype=np.float32)
    Rx = np.zeros([na, gsize, 9], dtype=np.float32)
    Rx2 = np.zeros([na, gsize, 9], dtype=np.float32)
    
    # see xyz convention in http://mathworld.wolfram.com/EulerAngles.html
    # D matrix
    Rz[:,0] = calpha
    Rz[:,1] = salpha
    Rz[:,2] = 0
    Rz[:,3] = -salpha
    Rz[:,4] = calpha
    Rz[:,5] = 0
    Rz[:,6] = 0
    Rz[:,7] = 0
    Rz[:,8] = 1
    
    # C matrix
    Ry[:,0] = cbeta
    Ry[:,1] = 0
    Ry[:,2] = sbeta
    Ry[:,3] = 0
    Ry[:,4] = 1
    Ry[:,5] = 0
    Ry[:,6] = -sbeta
    Ry[:,7] = 0
    Ry[:,8] = cbeta
    
    # B Matrix
    Rx[:,:,0] = 1
    Rx[:,:,1] = 0
    Rx[:,:,2] = 0
    Rx[:,:,3] = 0
    Rx[:,:,4] = np.cos(gamma)
    Rx[:,:,5] = np.sin(gamma)
    Rx[:,:,6] = 0
    Rx[:,:,7] = -np.sin(gamma)
    Rx[:,:,8] = np.cos(gamma)
    
    padding = 60
    Rx2[:,:,0] = 1
    Rx2[:,:,1] = 0
    Rx2[:,:,2] = 0
    Rx2[:,:,3] = 0
    Rx2[:,:,4] = np.cos(gamma+padding/180*np.pi)
    Rx2[:,:,5] = np.sin(gamma+padding/180*np.pi)
    Rx2[:,:,6] = 0
    Rx2[:,:,7] = -np.sin(gamma+padding/180*np.pi)
    Rx2[:,:,8] = np.cos(gamma+padding/180*np.pi)
    
    Rz = Rz[:,None].repeat(gsize,axis=1).reshape(na*gsize, 3,3)
    Ry = Ry[:,None].repeat(gsize,axis=1).reshape(na*gsize, 3,3)
    Rx = Rx.reshape(na*gsize,3,3)
    Rx2 = Rx2.reshape(na*gsize,3,3)
    
    # R = BCD
    Rxy = np.einsum('bij,bjh->bih', Rx, Ry)
    Rxy2 = np.einsum('bij,bjh->bih', Rx2, Ry)
    Rs1 = np.einsum('bij,bjh->bih', Rxy, Rz)
    Rs2 = np.einsum('bij,bjh->bih', Rxy2, Rz)
    
    z_val = (face_normals[:, -1])[:, None].repeat(gsize, axis=1).reshape(na*gsize, 1, 1)
    Rs = Rs1*(np.abs(z_val+0.79)<0.01)+Rs2*(np.abs(z_val+0.19)<0.01)+\
         Rs1*(np.abs(z_val-0.19)<0.01)+Rs2*(np.abs(z_val-0.79)<0.01)
    return Rs

def icosahedron_so3_trimesh(mesh_path, gsize=3, use_quats=False):
    # 20 faces, 12 vertices
    mesh = trimesh.load(mesh_path)
    mesh.fix_normals()
    face_idx = mesh.faces
    face_normals = mesh.face_normals
    fix_angle = np.arctan(face_normals[9, 2] / face_normals[9, 0])
    fix_rot = np.array([[np.cos(fix_angle),  0,  np.sin(fix_angle)],
                        [0,                  1,  0],
                        [-np.sin(fix_angle), 0, np.cos(fix_angle)]])
    
    na = face_normals.shape[0]
    Rs = get_so3_from_anchors_np(face_normals, gsize=gsize)
    Rs = np.einsum('bij,kj', Rs, Rs[29])
    R_adj = get_adjmatrix_trimesh(mesh, gsize)
    
    # Handle case where R_adj might be empty
    if R_adj.size > 0:
        grouped_R = np.take(Rs, R_adj, axis=0)
        relative_Rs = np.einsum('kjh,lh->kjl', grouped_R[0], Rs[0])
        ordered_R = np.einsum('kmj,bji->bkim', relative_Rs, Rs)
        canonical_R = None
        
        tiled_ordr = np.expand_dims(ordered_R,axis=2)
        diff_r = np.einsum('bkgij,chj->bkcih', tiled_ordr, Rs)
        trace = 0.5 * (np.einsum('bkgii->bkg', diff_r) - 1)
        trace_idx = np.argmax(trace,axis=2)
        
        reverse_Rs_idx = np.argmax(np.einsum('nij,mjk->nmji', Rs, Rs).sum(2).sum(2), axis=1)
        trace_idx = trace_idx[reverse_Rs_idx]
        
        use_idx = [2,3,6,9]
        new_trace_idx = np.zeros([trace_idx.shape[0], len(use_idx)], dtype=np.int32)
        for i in range(trace_idx.shape[0]):
            new_trace_idx[i] = trace_idx[i,use_idx]
    else:
        # Fallback when no adjacency information is available
        new_trace_idx = np.zeros([na, 4], dtype=np.int32)
        canonical_R = None
    
    if use_quats:
        Rs = sciR.from_matrix(Rs).as_quat()
    
    reverse_trace_idx = np.zeros_like(new_trace_idx)
    for i in range(new_trace_idx.shape[1]):
        for j in range(new_trace_idx.shape[0]):
            reverse_trace_idx[new_trace_idx[j,i], i] = j
    
    return Rs, new_trace_idx, canonical_R

def rotation_distance_np(r0, r1):
    '''
    tip: r1 is usally the anchors
    '''
    if r0.ndim == 3:
        bidx = np.zeros(r0.shape[0]).astype(np.int32)
        traces = np.zeros([r0.shape[0], r1.shape[0]]).astype(np.int32)
        for bi in range(r0.shape[0]):
            diff_r = np.matmul(r1, r0[bi].T)
            traces[bi] = np.einsum('bii->b', diff_r)
            bidx[bi] = np.argmax(traces[bi])
        return traces, bidx
    else:
        diff_r = np.matmul(np.transpose(r1,(0,2,1)), r0)
        traces = np.einsum('bii->b', diff_r)
        return traces, np.argmax(traces), diff_r

# Import torch for GPU functions
try:
    import torch
    
    def compute_rotation_matrix_from_quaternion(quaternion):
        def normalize_vector(v, return_mag=False):
            batch=v.shape[0]
            v_mag = torch.sqrt(v.pow(2).sum(1))
            v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
            v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
            v = v/v_mag
            if(return_mag==True):
                return v, v_mag[:,0]
            else:
                return v
        
        batch= quaternion.shape[0]
        quat = normalize_vector(quaternion).contiguous()
        qw = quat[...,0].contiguous().view(batch, 1)
        qx = quat[...,1].contiguous().view(batch, 1)
        qy = quat[...,2].contiguous().view(batch, 1)
        qz = quat[...,3].contiguous().view(batch, 1)
        
        xx = qx*qx
        yy = qy*qy
        zz = qz*qz
        xy = qx*qy
        xz = qx*qz
        yz = qy*qz
        xw = qx*qw
        yw = qy*qw
        zw = qz*qw
        
        row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1)
        row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1)
        row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1)
        matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1)
        return matrix

    def compute_rotation_matrix_from_euler_sin_cos(euler_sin_cos):
        batch=euler_sin_cos.shape[0]
        s1 = euler_sin_cos[:,0].view(batch,1)
        c1 = euler_sin_cos[:,1].view(batch,1)
        s2 = euler_sin_cos[:,2].view(batch,1)
        c2 = euler_sin_cos[:,3].view(batch,1)
        s3 = euler_sin_cos[:,4].view(batch,1)
        c3 = euler_sin_cos[:,5].view(batch,1)
        
        row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3)
        row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3)
        row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3)
        matrix = torch.cat((row1, row2, row3), 1)
        return matrix

    def compute_rotation_matrix_from_ortho6d(ortho6d):
        def normalize_vector(v, return_mag=False):
            batch=v.shape[0]
            v_mag = torch.sqrt(v.pow(2).sum(1))
            v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
            v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
            v = v/v_mag
            if(return_mag==True):
                return v, v_mag[:,0]
            else:
                return v
        
        def cross_product(u, v):
            batch = u.shape[0]
            i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
            j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
            k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
            out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)
            return out
        
        x_raw = ortho6d[:,0:3]
        y_raw = ortho6d[:,3:6]
        x = normalize_vector(x_raw)
        z = cross_product(x,y_raw)
        z = normalize_vector(z)
        y = cross_product(z,x)
        x = x.view(-1,3,1)
        y = y.view(-1,3,1)
        z = z.view(-1,3,1)
        matrix = torch.cat((x,y,z), 2)
        return matrix

    def so3_mean(Rs, weights=None):
        nb, na, _, _ = Rs.shape
        mask = torch.Tensor([[0,0,0],[0,0,0],[0,0,1]]).float().to(Rs.device)
        mask2 = torch.Tensor([[1,0,0],[0,1,0],[0,0,0]]).float().to(Rs.device)
        mask = mask[None].expand(nb, -1, -1).contiguous()
        mask2 = mask2[None].expand(nb, -1, -1).contiguous()
        
        if weights is None:
            weights = 1.0
        else:
            weights = weights[:,:,None,None]
        
        Ce = torch.sum(weights * Rs, dim=1)
        cu, cd, cv = torch.svd(Ce)
        cvT = cv.transpose(1,2).contiguous()
        dets = torch.det(torch.matmul(cu, cvT))
        D = mask * dets[:,None,None] + mask2
        return torch.einsum('bij,bjk,bkl->bil', cu, D, cvT)
        
except ImportError:
    print("Warning: PyTorch not available, GPU functions disabled")

def label_relative_rotation_np(anchors, T):
    T_from_anchors = np.einsum('abc,bj,ijk -> aick', anchors, T, anchors)
    label = np.argmax(np.einsum('abii->ab', T_from_anchors),axis=1)
    idxs = np.vstack([np.arange(label.shape[0]), label]).T
    R_target = T_from_anchors[idxs[:,0], idxs[:,1]]
    return R_target, label