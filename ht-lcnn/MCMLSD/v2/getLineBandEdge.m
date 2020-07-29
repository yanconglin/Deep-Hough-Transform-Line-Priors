function lineHolderEdge=getLineBandEdge(x1,y1,x2,y2,img,bandWidth,saliencyEdge)
v=[x1,y1]-[x2,y2];
v=v/norm(v);
vperp=[v(2),-v(1)];
vn = ceil(sqrt((x1-x2)^2+(y1-y2)^2));
sEdge=getLineTPEdge(x1,y1,x2,y2,img,vn,saliencyEdge);
lineHolderEdge=zeros(vn,bandWidth*2+1);
lineHolderEdge(:,bandWidth+1)=sEdge;
for j=1:bandWidth
    sEdge=getLineTPEdge(x1-j*vperp(1),y1-j*vperp(2),x2-j*vperp(1),y2-j*vperp(2),img,vn,saliencyEdge);
    lineHolderEdge(:,bandWidth+1-j)=sEdge;
    
    sEdge=getLineTPEdge(x1+j*vperp(1),y1+j*vperp(2),x2+j*vperp(1),y2+j*vperp(2),img,vn,saliencyEdge);
    lineHolderEdge(:,bandWidth+1+j)=sEdge;
end

end