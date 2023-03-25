library("KEGGREST")
# 读取文件的内容
my_lines <- readLines("D:\\KEGG\\pathway.txt")
# 计算文件的行数
num_lines <- length(my_lines)
# 输出结果
cat("The file contains", num_lines, "lines.")
for (i in 1:num_lines) {
  path_name <- my_lines[i]
  keggGet(path_name)
  gs<-keggGet(paste(path_name))
  #获取通路中gene信息
  gs[[1]]$GENE
#查找所有基因
  #genes<-unlist(lapply(gs[[1]]$GENE,function(x) strsplit(x,';')))
  genes <- unlist(lapply(gs[[1]]$GENE, function(x) {
  s <- strsplit(trimws(x), "; ")[[1]]
  if (length(s) == 2) {
    return(s)
  } else {
    print(paste0("Skipping element: ", x))
    return(NULL)
  }
}))
  genelist <- genes[1:length(genes)%%2 ==1]
  genelist <- data.frame(genelist)
#把结果写入表格中
path<-paste("D:\\KEGG\\gene_csv\\" ,path_name,".csv", sep ="")
write.table(genelist, path,
row.names=FALSE,col.names=TRUE,sep=",")
}

# library("KEGGREST")
#
# setwd("E:\\xiong\\Bioinfomatics\\DriverGenes\\WorkingSpace\\MyDriveGenes\\RawData\\Associations\\KEGG")
#
# pathways <- read.table("kegg_pathway_hsa.txt", sep='\t')
#
# # Calculate the number of pathways
# num_pathways <- dim(pathways)[1] # 352
#
# count_null <- 0
#
# for (i in 1:num_pathways) {
#   gene_list <- list()
#   # Iterate each pathway
#   pathway <- pathways$V1[i]
#   # Get pathway from KEGG
#   pathway_info <- keggGet(pathway)
#   # Get gene info in the pathway -- Attention: might be null
#   gene_info <- pathway_info[[1]]$GENE
#   # Get all the genes in the pathway
#   if(is.null(gene_info)){
#     count_null <- count_null + 1
#     cat("Find", count_null, "empty pathways!")
#   }else{
#     for(i in 1:length(gene_info)){
#       if(i %% 2 == 0){
#         gene_list <- c(gene_list, strsplit(gene_info[i], ';')[[1]][1])
#       }
#     }
#     # Save the gene of each pathway in the format of csv
#     gene_df <- data.frame(gene_list)
#     save_path <- paste("E:\\xiong\\Bioinfomatics\\DriverGenes\\WorkingSpace\\MyDriveGenes\\RawData\\Associations\\KEGG\\", pathway, '.csv', sep="")
#     write.table(gene_list, file=save_path, col.names = FALSE, row.names = FALSE, sep=",")
#   }
# }

