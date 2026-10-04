"""Complete simplex searches using bounding-volume hierarchies."""
import numpy as np
from numba import njit


def build_hierarchy(lower, upper, leaf_size=24):
    """Median-split cell boxes; leaves retain original cell indices."""
    centers=.5*(lower+upper)
    boxes_low,boxes_high,left,right,start,count,ordered=[],[],[],[],[],[],[]
    max_depth=0

    def split(indices,depth):
        nonlocal max_depth
        max_depth=max(max_depth,depth)
        node=len(left)
        boxes_low.append(lower[indices].min(axis=0))
        boxes_high.append(upper[indices].max(axis=0))
        left.append(-1);right.append(-1);start.append(-1);count.append(0)
        if len(indices)<=leaf_size:
            start[node]=len(ordered);count[node]=len(indices)
            ordered.extend(indices)
        else:
            axis=int(np.argmax(np.ptp(centers[indices],axis=0)))
            mid=len(indices)//2
            partition=np.argpartition(centers[indices,axis],mid)
            left[node]=split(indices[partition[:mid]],depth+1)
            right[node]=split(indices[partition[mid:]],depth+1)
        return node

    split(np.arange(len(lower)),0)
    return (np.asarray(boxes_low),np.asarray(boxes_high),np.asarray(left),np.asarray(right),
            np.asarray(start),np.asarray(count),np.asarray(ordered),max_depth+2)


@njit(cache=True)
def locate_many(points,origin,inverse,cell_lower,cell_upper,box_lower,box_upper,
                left,right,start,count,ordered,stack_size):
    """Search all enclosing boxes, including nonadjacent overlaps.

    Status 0: found; 1: outside; 2: more than one strict interior.
    Shared faces choose the smallest original cell index deterministically.
    """
    n,dimension=points.shape
    selected=np.full(n,-1,np.int64)
    weights=np.zeros((n,dimension+1))
    status=np.ones(n,np.int8)
    stack=np.empty(stack_size,np.int64)
    bary=np.empty(dimension+1)
    for p in range(n):
        top=1;stack[0]=0
        interiors=0
        while top:
            top-=1;node=stack[top]
            hit=True
            for d in range(dimension):
                if points[p,d]<box_lower[node,d]-1e-10 or points[p,d]>box_upper[node,d]+1e-10:
                    hit=False;break
            if not hit:
                continue
            if left[node]>=0:
                stack[top]=left[node];stack[top+1]=right[node];top+=2
                continue
            for k in range(start[node],start[node]+count[node]):
                cell=ordered[k]
                hit=True
                for d in range(dimension):
                    if points[p,d]<cell_lower[cell,d]-1e-10 or points[p,d]>cell_upper[cell,d]+1e-10:
                        hit=False;break
                if not hit:
                    continue
                bary[0]=1.
                for d in range(dimension):
                    bary[d+1]=0.
                    for e in range(dimension):
                        bary[d+1]+=inverse[cell,d,e]*(points[p,e]-origin[cell,e])
                    bary[0]-=bary[d+1]
                hit=True;interior=True
                for d in range(dimension+1):
                    if bary[d]<-1e-9 or bary[d]>1.+1e-9:
                        hit=False;break
                    if bary[d]<=1e-7:
                        interior=False
                if not hit:
                    continue
                if interior:
                    interiors+=1
                    if interiors>1:
                        status[p]=2
                        break
                if interior or (interiors==0 and (selected[p]<0 or cell<selected[p])):
                    selected[p]=cell
                    for d in range(dimension+1):
                        weights[p,d]=bary[d]
                    status[p]=0
            if status[p]==2:
                break
    return selected,weights,status
