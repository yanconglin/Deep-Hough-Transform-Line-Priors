function sEdge=getLineTPEdge(x1,y1,x2,y2,img,vn,saliencyEdge)
global edgeMap;

[height,width,~]=size(img);


cur_edge = [linspace(y1,y2,vn).', linspace(x1,x2,vn).'];
pixelLine=round(cur_edge);
pixelLine(pixelLine(:,2)<=0,2)=1;
pixelLine(pixelLine(:,1)<=0,1)=1;
pixelLine(pixelLine(:,1)>height,1)=height;
pixelLine(pixelLine(:,2)>width,2)=width;
% [pixelLineC,~,~] = unique(pixelLine,'rows','legacy');
linearInd = sub2ind([height,width], pixelLine(:,1), pixelLine(:,2));
sEdge= saliencyEdge(linearInd).*logical(edgeMap.edge(linearInd));
end