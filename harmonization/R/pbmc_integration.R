# Code adapted from https://github.com/satijalab/Integration2019/blob/master/analysis_code/citeseq/cite_cross_validations.R
# loaded data is log normalized RNA already selected for HVGs concatenated to log transformed proteins

library(Seurat)
set.seed(1234)

# setwd("/data/yosef2/users/adamgayoso/projects/totalVI_journal")
setwd("~/Google Drive/Berkeley/Yosef_lab/totalVI_journal")

umis_sln_d1 = read.csv('data/raw_data/pbmc10k_harmo_rna_pro.csv.gz', row.names = 1)
umis_sln_d2 = read.csv('data/raw_data/pbmc5k_harmo_rna_pro.csv.gz', row.names = 1)

sln_d1 <- CreateSeuratObject(counts = umis_sln_d1)

DefaultAssay(object = sln_d1) <- "RNA"
sln_d1 <- ScaleData(object = sln_d1)
features <- rownames(sln_d1)

sln_d2 <- CreateSeuratObject(counts = umis_sln_d2)
DefaultAssay(object = sln_d2) <- "RNA"
sln_d2 <- ScaleData(object = sln_d2)

sln.list <- list("d1" = sln_d1, "d2" = sln_d2)
sln.anchors <- FindIntegrationAnchors(object.list = sln.list, dims = 1:30, anchor.features = features, normalization.method = "LogNormalize")

sln.integrated <- IntegrateData(anchorset = sln.anchors, dims = 1:30)

integrated.data <- GetAssayData(
  object = sln.integrated,
  assay = 'integrated',
  slot = 'data'
)

write.csv(integrated.data, file=gzfile("harmonization/seurat_harmo_results/seurat_pbmc_integrated.csv.gz"))
