import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import ezdxf
from ezdxf import colors
from ezdxf.enums import TextEntityAlignment

class panel:
    '''A class used to generate interconnected wire gradient panels.

This class takes a list of lists, which represent individual current carrying
loops in a gradient coil panel. It finds a non-intersecting network of shortest
paths between all of the points specified with a Delaunay triangulation.

The first loop in the list (loops[0]) is treated as the connection to the
outside world. We find the shortest edge from the Delaunay triangulation that
connects this loop to a point not in this loop.

This edge is added to the "connections" list, both as coordinates (connections)
and as point indices (connection_index).

The loop that was connected to is added to the "energized" list, and the process
repeats.
 - Find the shortest edge containing a single point in "energized" list
 - Calculate the "connection"
 - Update the "energized" list
 - Repeat as long as any loops specified are not "energized"

This procedure runs when the class is initiated.

To generate a DXF file, use the make_dxf() method. This method adds helper marks
to show the winding direction: ticks on the right as you move along the wire.

By default it will  draw 4 ticks for each loop. If a loop needs more (or fewer),
use the list instantiated as self.helpers
For example to make loops[3] have 12 ticks while the rest have the default of 4:

gradpanel = panel(wirelist)
gradpanel.helpers[3]=12
gradpanel.make_dxf()  # note: this function has a bunch of tunable parameters

    '''
    def __init__(self, loops):
        self.wires = loops
        
        # Assume that the first "loop" in list is connected to the outside world
        self.energized = np.zeros(len(loops))
        self.energized[0] = True
        
        self.N = len(loops)
        self.connection_index = []
        self.connections = []
        
        self.loop_start_n = np.zeros(len(loops),dtype='uint32')
        # Override how many direction helpers a particular loop should have
        self.helpers = np.zeros_like(self.energized)
        
        for i, wire in enumerate(loops):
            if i==0:
                points=wire.reshape((2,-1)).T
                self.loop_start_n[2*i]=0
            else:
                points = np.concatenate((points, wire.reshape((2,-1)).T))
                self.loop_start_n[i]=self.loop_start_n[i-1]+last_loop
            last_loop = np.max((len(wire),len(wire.T)))
            
        self.points = points
        self.tri=Delaunay(points)
        
        self.link_loops()
    
    def get_neighbor_vertex_ids_from_vertex_id(self,vertex_id):
        index_pointers, indices = self.tri.vertex_neighbor_vertices
        result_ids = indices[index_pointers[vertex_id]:index_pointers[vertex_id + 1]]
        return result_ids
    
    def get_unconnected_neighbors_to_energized(self):
        # append all hooked up points
        # then filter neighbors to exclude pairs within loop
        hooked_up_points = np.array([])
        
        for i,(connected,n) in enumerate(zip(self.energized,self.loop_start_n)):
            if connected:  # only search connections to loops that are energized
                for j,coord in enumerate(self.wires[i].T):
                    index = n+j
                    neighbors = self.get_neighbor_vertex_ids_from_vertex_id(index)
                    matchpair = np.stack(([index]*len(neighbors),neighbors))
                    if index==0:
                        hooked_up_points = matchpair
                    else:
                        hooked_up_points=np.concatenate(
                            (hooked_up_points,matchpair),
                            axis=1)
        # matches = np.unique(hooked_up_points[1,:])
        # sources = hooked_up_points[0,:]
        # map matches to boolean filter from 0th row
        mask = np.logical_not(np.isin(
            hooked_up_points[1,:],
            hooked_up_points[0,:]))
        self.branches=hooked_up_points[:,mask].astype('uint32').T
    
    def attach_shortest_branch(self):
        # for pair in self.branches:
        #     # points[n[0],0],points[n[1],0]),(points[n[0],1],points[n[1],1]
        #     deltar = self.points[pair[0],:]-points[pair[1],:]
        #     branchlengths.append(np.dot(deltar,deltar))
        
        lengths = np.linalg.norm(self.points[self.branches[:,0],:]-
                                 self.points[self.branches[:,1],:],
                                 axis=1)
        short_index = np.array(np.nonzero(lengths == np.min(lengths))).ravel()[0]
        self.connection_index.append(self.branches[short_index,:])
        
        self.connections.append(
            np.array((
                self.points[self.branches[short_index,0],:],
                self.points[self.branches[short_index,1],:]
            )))
    
    def energize_connection(self):
        # Find the loop connected to the far end of the most recent connection
        # Set its energized to True
        hookup = self.connection_index[-1][-1]
        new_wire = np.where(False==np.array(self.loop_start_n>hookup))[0][-1]
        self.energized[new_wire]=True
    
    def link_loops(self):
        while not np.all(self.energized):
            self.get_unconnected_neighbors_to_energized()
            self.attach_shortest_branch()
            self.energize_connection()
    
    def make_dxf(self,filename='tripanel',
        helpers=True, gap=.1, length = .1, stride=4,
        textstring="", textcenter = np.array((0,0)), textheight = 0.01):
        # Create a new DXF document.
        doc = ezdxf.new(dxfversion="R2010")
        unified = doc.layers.add("merged",color=255)
        loop_layer = doc.layers.add("loops",color=1)
        interconnects = doc.layers.add("connections",color=2)   
        helper_layer = doc.layers.add("direction",color=3)
        text_layer = doc.layers.add("text",color=4)

        msp = doc.modelspace()
        
        for wire in self.wires:
            for layer in ("loops","merged"):
                msp.add_lwpolyline(wire.T, dxfattribs={"layer": layer})
        
             
        for pair in self.connections:
            for layer in ("connections","merged"):
                msp.add_line(pair[0,:],pair[1,:],dxfattribs={"layer": layer})
            
        if helpers:
            for wire,n_helpers in zip(self.wires,self.helpers):
                maxn =len(wire[0,:])
                if n_helpers:
                    stepsize = maxn//n_helpers
                    step = n_helpers
                else:
                    stepsize = maxn//stride
                    step = stride

                for i in range(step):
                    p0 = wire[:,stepsize*i]
                    p1 = wire[:,stepsize*i+1]
                    r = p1-p0

                    outside = np.flip(r).copy()
                    outside[1]*=-1
                    outside/=np.linalg.norm(outside)
                    
                    r0 = p0+outside*gap
                    r1 = p0+outside*(gap+length)

                    for layer in ("direction","merged"):
                        msp.add_lwpolyline((r0,r1), dxfattribs={"layer": layer})
        if textstring:
            msp.add_text(
                textstring, 
                height = textheight,
                dxfattribs={
                    "layer": "text"
                }).set_placement(textcenter, align=TextEntityAlignment.CENTER)

        # Save the DXF document.
        doc.saveas(filename+'.dxf')
