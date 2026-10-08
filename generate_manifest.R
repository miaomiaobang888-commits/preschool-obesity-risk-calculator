# 在完整自检通过后生成真实依赖清单；不编造manifest，不自动发布。
if(!requireNamespace("rsconnect",quietly=TRUE)) install.packages("rsconnect",repos="https://cloud.r-project.org")
source("deployment_self_test.R",encoding="UTF-8")
# 只打包白名单文件。整个开发结果目录、SHAP个体RDS、截图、脚本中的本机路径不上传。
app_files<-c("app.R","data/XGBoost.ubj","data/XGB_web_config.rds",
             "data/XGBoost_SHAP_web_summary.rds","www/XGBoost_SHAP_feature_importance.png",
             "www/XGBoost_SHAP_beeswarm.png")
stopifnot(all(file.exists(app_files)))
rsconnect::writeManifest(appDir=getwd(),appFiles=app_files,appPrimaryDoc="app.R",appMode="shiny")
stopifnot(file.exists("manifest.json"))
cat("已根据本机真实依赖生成manifest.json。更改模型/配置/依赖/图片后请重新运行本脚本。\n")
