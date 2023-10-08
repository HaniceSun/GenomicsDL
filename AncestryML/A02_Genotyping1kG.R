library(bigsnpr)
library(ggplot2)

args = commandArgs(trailingOnly=TRUE)

bedfile = args[1]  
plink2 = '/oak/stanford/groups/agloyn/hansun/MySoft/plink/plink2'
nCores=8

## Relatedness
rel <- snp_plinkKINGQC(
  plink2.path = plink2,
  bedfile.in = bedfile,
  thr.king = 2^-4.5,
  make.bed = FALSE,
  ncores = nCores
)
print(rel)

## compute PCA without using the related individuals
obj.bed <- bed(bedfile)

### rel
#ind.rel <- match(c(rel$IID1, rel$IID2), obj.bed$fam$sample.ID)

### rel + 1000G/HGDP
ts = obj.bed$fam$sample.ID[!grepl('^HG|^NA', obj.bed$fam$sample.ID)]
ind.rel <- match(c(rel$IID1, rel$IID2, ts), obj.bed$fam$sample.ID)
ind.norel <- rows_along(obj.bed)[-ind.rel]

obj.svd <- bed_autoSVD(obj.bed, ind.row = ind.norel, k = 20, ncores = nCores)

## Outlier sample detection

prob <- bigutilsr::prob_dist(obj.svd$u, ncores = nCores)
S <- prob$dist.self / sqrt(prob$dist.nn)

pdf(gsub('.bed', '.pdf', bedfile))

ggplot() +
  geom_histogram(aes(S), color = "#000000", fill = "#000000", alpha = 0.5) +
  scale_x_continuous(breaks = 0:5 / 5, limits = c(0, NA)) +
  scale_y_sqrt(breaks = c(10, 100, 500)) +
  theme_bigstatsr() +
  labs(x = "Statistic of outlierness", y = "Frequency (sqrt-scale)")

### Outlier threshold based on histogram
OutlierThreshold = 0.5 

#plot_grid(plotlist = lapply(7:10, function(k) {
#  plot(obj.svd, type = "scores", scores = 2 * k - 1:0, coeff = 0.6) +
#    aes(color = S) +
#    scale_colour_viridis_c()
#}), scale = 0.95)

plot_grid(plotlist = lapply(7:10, function(k) {
  plot(obj.svd, type = "scores", scores = 2 * k - 1:0, coeff = 0.6) +
    aes(color = S > OutlierThreshold) + 
    scale_colour_viridis_d()
}), scale = 0.95)

## PCA without outlier

ind.row <- ind.norel[S < OutlierThreshold]
ind.col <- attr(obj.svd, "subset")
obj.svd2 <- bed_autoSVD(obj.bed, ind.row = ind.row, ind.col = ind.col, thr.r2 = NA, k = 20, ncores = nCores)

## Verification

plot(obj.svd2)

plot(obj.svd2, type = "loadings", loadings = 1:20, coeff = 0.4)

plot(obj.svd2, type = "scores", scores = 1:20, coeff = 0.4)

## Project remaining individuals

PCs <- matrix(NA, nrow(obj.bed), ncol(obj.svd2$u))
PCs[ind.row, ] <- predict(obj.svd2)

proj <- bed_projectSelfPCA(obj.svd2, obj.bed, ind.row = rows_along(obj.bed)[-ind.row], ncores = 1)
PCs[-ind.row, ] <- proj$OADP_proj

plot(PCs[ind.row, 7:8], pch = 20, xlab = "PC7", ylab = "PC8")
points(PCs[-ind.row, 7:8], pch = 20, col = "blue")

dev.off()


annot = read.table('/oak/stanford/groups/agloyn/hansun/Data/1000G/1000G_30x_hg38/1000G_UnrelatedSamples_2504S.txt', header=T)
PCs_df = data.frame(PCs)
PCs_df$Sample = obj.bed$fam$sample.ID
PCs_df$ObsPre = NA
PCs_df$ObsPre[ind.row] = 'Observed'
PCs_df$ObsPre[-ind.row] = 'ToBePredicted'
PCs_df = merge(PCs_df, annot, by.x='Sample', by.y='SampleID', all.x=T)

#PCs_df = grepl('^HG|^NA', dfs_df$Sample)
write.table(PCs_df, gsub('.bed', '_PCs.txt', bedfile), row.names=F, col.names=T, sep='\t', quote=F)

save.image(gsub('.bed', '.rda', bedfile))

