# ============================================================================
# 锁定XGBoost风险计算器：本文件不训练、不调参、不重新估计预处理或阈值。
# 使用：先运行prepare_xgb_web_bundle.R，然后在本目录运行shiny::runApp()。
# 部署依赖在本地准备，不在云端启动时安装R包。
# ============================================================================
library(shiny)
library(bslib)
library(xgboost)
library(shapviz)
library(ggplot2)
library(dplyr)
library(DT)
library(scales)
library(jsonlite)

assert <- function(ok, ...) if (!isTRUE(ok)) stop(paste0(..., collapse=""), call.=FALSE)
exact_names <- function(actual, expected, label) {
  if (!identical(actual, expected)) {
    message(label, "\nMissing features: ", paste(setdiff(expected, actual), collapse=", "),
            "\nExtra features: ", paste(setdiff(actual, expected), collapse=", "))
    n <- max(length(actual), length(expected))
    print(data.frame(Position=seq_len(n), Expected=expected[seq_len(n)],
                     Actual=actual[seq_len(n)]))
    stop(paste(label, "名称或顺序不一致；Order differences见上表。"), call.=FALSE)
  }
  invisible(TRUE)
}
get_features <- function(model) {
  nms <- tryCatch(xgboost::getinfo(model, "feature_name"), error=function(e) NULL)
  if (is.null(nms) || !length(nms))
    nms <- tryCatch(model$feature_names, error=function(e) NULL)
  assert(is.character(nms) && length(nms)>0 && !anyNA(nms) && !anyDuplicated(nms),
         "模型没有完整的feature names，禁止凭经验补写。")
  nms
}
# 输入必须是已经定义的临床编码：二分类0/1、进食速度原水平1/2/3。
# 这里不接收原数据库纵容型1/2编码，避免重复翻转No/Yes。
make_web_matrix <- function(values, config, enforce_bounds=TRUE) {
  exact_names(names(values), config$predictor_vars, "网页9项输入")
  assert(all(lengths(values)>0L) && length(unique(lengths(values)))==1L, "请完整填写全部9项输入。")
  if (is.list(values)) values <- as.data.frame(values, check.names=FALSE,
                                              stringsAsFactors=FALSE)
  assert(nrow(values)>0L, "输入为空。")
  for (v in config$predictor_vars) {
    s <- config$schema[[v]]; x <- as.character(values[[v]])
    assert(!anyNA(x) && all(nzchar(trimws(x))), v, "为必填项。")
    if (s$type=="categorical") {
      assert(all(x %in% s$levels), v, "必须使用锁定的类别。")
      values[[v]] <- x
    } else {
      a <- suppressWarnings(as.numeric(x))
      assert(all(is.finite(a)), v, "必须填写有限数值。")
      if (s$type=="binary") assert(all(a %in% c(0,1)), v, "只接受0/1。")
      b <- config$input_bounds[[v]]
      if (enforce_bounds && length(b)==2L)
        assert(all(a>=b[1] & a<=b[2]), v, "允许范围为", b[1], "至", b[2], "。")
      values[[v]] <- a
    }
  }
  # 按保存的feature map逐列构造；不猜测列名、不根据单例重新model.matrix。
  fm <- config$feature_map
  raw <- matrix(0, nrow(values), nrow(fm), dimnames=list(NULL, fm$Feature))
  for (k in seq_len(nrow(fm))) {
    x <- values[[fm$Variable[k]]]
    raw[,k] <- if (fm$Kind[k]=="dummy") as.numeric(x==fm$Level[k]) else as.numeric(x)
  }
  exact_names(colnames(raw), config$feature_names, "编码矩阵")
  if (identical(config$matrix_transform, "training_center_scale_once")) {
    exact_names(names(config$center), colnames(raw), "训练center")
    exact_names(names(config$scale), colnames(raw), "训练scale")
    assert(all(is.finite(config$center)) && all(is.finite(config$scale)) &&
             all(config$scale>0), "标准化配置非法。")
    raw <- sweep(sweep(raw, 2, config$center, "-"), 2, config$scale, "/")
  } else if (!identical(config$matrix_transform, "none")) stop("未知矩阵变换。")
  assert(all(is.finite(raw)), "预测矩阵含非法值。")
  raw
}
# 兼容BIAS、(Intercept)和无列名输出；最后一列为官方接口定义的基线。
# 只有贡献列顺序、输出维度、加性检查同时通过，才用于网页解释。
split_tree_shap <- function(raw, features, margin, tolerance=1e-5) {
  p <- length(features)
  assert(is.matrix(raw) && is.numeric(raw) &&
           identical(dim(raw), c(as.integer(length(margin)), as.integer(p+1L))) &&
           all(is.finite(raw)), "TreeSHAP输出不是n×(p+1)有限数值矩阵。")
  if (!is.null(colnames(raw))) {
    exact_names(colnames(raw)[seq_len(p)], features, "TreeSHAP贡献列")
    assert(!isTRUE(colnames(raw)[p+1L] %in% features), "基线列与特征重名。")
  }
  S <- raw[,seq_len(p),drop=FALSE]; colnames(S) <- features
  bias <- as.numeric(raw[,p+1L]); err <- abs(rowSums(S)+bias-margin)
  assert(max(err)<=tolerance, "TreeSHAP重建margin失败；最大误差=", max(err))
  list(S=S, bias=bias, error=err)
}
aggregate_shap <- function(S, config) {
  out <- matrix(0, nrow(S), length(config$predictor_vars),
                dimnames=list(NULL, config$predictor_vars))
  exact_names(colnames(S), config$feature_names, "归并前贡献")
  for (v in config$predictor_vars) {
    f <- config$feature_map$Feature[config$feature_map$Variable==v]
    assert(length(f)>0L, "归并映射缺失：", v)
    out[,v] <- rowSums(S[,f,drop=FALSE]) # 有符号逐例相加，不能先取绝对值
  }
  out
}
clinical_display <- function(values, config, english=FALSE) {
  d <- as.data.frame(values, check.names=FALSE, stringsAsFactors=FALSE)
  for (v in config$predictor_vars) {
    s <- config$schema[[v]]
    if (s$type=="binary") d[[v]] <- factor(as.character(d[[v]]), levels=c("0","1"),
                                                labels=if(english) c("No","Yes") else c("否","是"))
    if (s$type=="categorical") d[[v]] <- factor(as.character(d[[v]]), levels=s$levels,
                                    labels=if(english && !config$category_labels_confirmed) paste("Category",s$levels) else unname(config$eating_speed_labels[s$levels]))
  }
  d
}
predict_one <- function(values, engine) {
  cfg <- engine$config; x <- make_web_matrix(values, cfg)
  exact_names(colnames(x), engine$features, "模型输入")
  dm <- xgboost::xgb.DMatrix(x, nthread=1L)
  p <- as.numeric(predict(engine$model, dm))
  assert(length(p)==1L && is.finite(p) && p>=0 && p<=1, "概率预测失败。")
  # 风险结果与解释分开处理：SHAP失败不伪造贡献，也不抹去可用概率。
  base <- list(probability=p, cutoff=cfg$locked_cutoff,
               predicted_class=as.integer(p>=cfg$locked_cutoff), values=values,
               timestamp=format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
               shap_ok=FALSE, error=NULL)
  tryCatch({
    margin <- as.numeric(predict(engine$model, dm, outputmargin=TRUE))
    assert(length(margin)==1 && is.finite(margin), "margin非法。")
    assert(abs(plogis(margin)-p)<=1e-5, "margin与概率不一致。")
    raw <- predict(engine$model, dm, predcontrib=TRUE, approxcontrib=FALSE)
    z <- split_tree_shap(raw, engine$features, margin)
    S <- aggregate_shap(z$S, cfg)
    err <- abs(rowSums(S)+z$bias-margin)
    assert(max(err)<=1e-5, "9变量归并重建误差超过1e-5。")
    disp <- clinical_display(values, cfg, english=TRUE)
    colnames(S) <- unname(cfg$feature_label_map[colnames(S)])
    names(disp) <- unname(cfg$feature_label_map[names(disp)])
    sv <- shapviz::shapviz(S, X=disp, baseline=z$bias[1])
    base$shap_ok <- TRUE; base$shap <- setNames(as.numeric(S[1,]), cfg$predictor_vars)
    base$bias <- z$bias[1]; base$margin <- margin; base$sv <- sv
    base$reconstruction_error <- max(err); base$link_error <- abs(plogis(margin)-p)
    base
  }, error=function(e) {
    warning("个体SHAP不可用：", conditionMessage(e), call.=FALSE)
    base$error <- "解释校验未通过，本次仅显示预测概率；请联系研究人员检查。"
    base
  })
}
# 只读启动：核对模型身份、目标函数、特征与部署用合成测试样例。
load_engine <- function(root=".") {
  mf <- file.path(root,"data","XGBoost.ubj")
  cf <- file.path(root,"data","XGB_web_config.rds")
  assert(file.exists(mf) && file.exists(cf), "缺少模型/网页配置；请先运行prepare_xgb_web_bundle.R。")
  cfg <- readRDS(cf)
  assert(identical(cfg$format_version, "1.0"), "不支持的网页配置版本。")
  assert(identical(unname(tools::md5sum(mf)), cfg$model_md5), "模型与网页配置不属于同一文件版本。")
  model <- xgboost::xgb.load(mf)
  assert(inherits(model,"xgb.Booster"), "模型类型错误。")
  mc <- xgboost::xgb.config(model)
  if (is.character(mc)) mc <- jsonlite::fromJSON(mc, simplifyVector=FALSE)
  assert(identical(mc$learner$objective$name, "binary:logistic"), "仅支持阳性为1的binary:logistic模型。")
  features <- get_features(model)
  exact_names(features, cfg$feature_names, "模型feature names")
  exact_names(cfg$feature_map$Feature, features, "保存的feature map")
  assert(length(cfg$predictor_vars)==9L && !anyDuplicated(cfg$predictor_vars) &&
           setequal(cfg$feature_map$Variable,cfg$predictor_vars), "原始9因子映射错误。")
  assert(length(cfg$locked_cutoff)==1L && is.finite(cfg$locked_cutoff) &&
           cfg$locked_cutoff>0 && cfg$locked_cutoff<1, "cutoff非法。")
  message(sprintf("Locked XGBoost cutoff = %.12f",cfg$locked_cutoff))
  if (abs(cfg$locked_cutoff-0.254002)>1e-5) warning("cutoff与既往约值不同，请核对批次。")
  if (identical(cfg$category_labels_confirmed,FALSE))
    warning("进食速度临床标签未确认；网页显示类别1/2/3，请在准备脚本填写数据字典标签。")
  if (any(vapply(cfg$input_bounds[c("挑食","过饱响应","情绪性进食")],is.null,logical(1))))
    warning("行为评分理论范围未确认；当前只检查有限数值。正式开放前请按量表字典配置范围。")
  e <- list(model=model, config=cfg, features=features, global=NULL)
  tc <- cfg$deployment_test
  xp <- make_web_matrix(tc$values,cfg)
  tp <- as.numeric(predict(model,xgboost::xgb.DMatrix(xp,nthread=1L)))
  error <- abs(tp-tc$expected_probability)
  assert(length(error)==1L && is.finite(error) && error<1e-8,
         "部署测试样例概率不一致；请核对包版本、预处理与模型，误差=", error)
  e$selftest_error <- error
  message(sprintf("Synthetic deployment self-test passed; absolute error=%.3g",error))
  sf <- file.path(root,"data","XGBoost_SHAP_web_summary.rds")
  if (file.exists(sf)) e$global <- tryCatch({
    z <- readRDS(sf); g <- z$global_importance
    assert(identical(z$model_md5,cfg$model_md5),"全局SHAP摘要模型哈希不匹配。")
    assert(is.data.frame(g) && all(c("Variable","Mean_abs_SHAP") %in% names(g)) &&
             nrow(g)==9 && !anyDuplicated(g$Variable) &&
             setequal(g$Variable,cfg$predictor_vars) && all(is.finite(g$Mean_abs_SHAP)) &&
             all(g$Mean_abs_SHAP>=0), "全局SHAP摘要格式异常。")
    z$global_importance <- g[order(-g$Mean_abs_SHAP),,drop=FALSE]
    if(z$global_importance$Variable[1]!="母亲BMI") warning("全局SHAP首位不同于既往结果，请核对批次。")
    z
  },error=function(e){ warning(conditionMessage(e)); NULL })
  for(f in c("XGBoost_SHAP_feature_importance.png","XGBoost_SHAP_beeswarm.png"))
    message("Global image ",f,": ",file.exists(file.path(root,"www",f)))
  e
}
# 主程序初始化；失败时给Console完整原因，不启动错误模型。
engine <- tryCatch(load_engine(), error=function(e)
  stop("XGBoost计算器启动失败：",conditionMessage(e),call.=FALSE))
cfg <- engine$config
input_ids <- sprintf("v%02d",seq_along(cfg$predictor_vars))
plot_theme <- theme_minimal(base_size=12,base_family="sans") +
  theme(panel.grid.minor=element_blank(), plot.title=element_text(color="#1F4E78",face="bold"))
shap_note <- "SHAP位于模型margin（log-odds）尺度。正值推动预测风险升高，负值推动风险降低；不是概率百分点、OR或因果效应。"
# 数值输入仅采用已确认范围；未确认的量表上下限不编造为1–5。
input_control <- function(j) {
  v <- cfg$predictor_vars[j]; s <- cfg$schema[[v]]; default <- cfg$defaults[[v]]
  if (s$type=="binary") return(selectInput(input_ids[j],v,choices=c("否"="0","是"="1"),selected=as.character(default)))
  if (s$type=="categorical") return(selectInput(input_ids[j],v,
          choices=setNames(s$levels,unname(cfg$eating_speed_labels[s$levels])), selected=as.character(default)))
  a <- list(inputId=input_ids[j],label=if(v %in% c("父亲BMI","母亲BMI")) paste0(v,"（kg/m²）") else v,value=as.numeric(default),step=0.1)
  b <- cfg$input_bounds[[v]]
  if(length(b)==2L) {a$min<-b[1];a$max<-b[2]}
  do.call(numericInput,a)
}
css <- "
:root{--navy:#1F4E78;--blue:#0072B2;--muted:#6C757D;}
body{background:#fff;color:#263747;font-family:Arial,'Microsoft YaHei',sans-serif;}
.app-wrap{max-width:1920px;margin:auto;padding:20px 28px;}
.app-title{font-size:27px;font-weight:700;color:var(--navy);margin:0 0 5px;}
.en-title{font-size:17px;color:var(--navy);margin-bottom:3px;}
.sub-title,.muted{color:var(--muted);font-size:13px;}
.header-line{border-bottom:2px solid var(--navy);padding-bottom:14px;margin-bottom:14px;}
.intro{font-size:14px;line-height:1.7;margin:12px 0;}
.app-grid{display:grid;grid-template-columns:minmax(270px,29%) minmax(0,1fr);gap:20px;align-items:start;}
.input-panel{border:1px solid #dfe5eb;border-radius:8px;padding:16px;background:#f8fafc;}
.input-panel h3{font-size:19px;color:var(--navy);margin-bottom:12px;}
.input-panel .form-group{margin-bottom:9px;}.input-panel label{font-size:13px;margin-bottom:4px;}
.input-panel input,.input-panel select{font-size:14px;min-height:34px;}
.btn-primary{background:var(--navy);border-color:var(--navy);}.btn-row{display:flex;gap:8px;margin:14px 0;}
.card{border:1px solid #dfe5eb;box-shadow:none;}.nav-link{font-size:13px;}
.probability{font-size:42px;line-height:1.15;font-weight:700;color:var(--navy);}
.risk-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:15px 0;}
.metric-block{padding:16px;border:1px solid #dfe5eb;border-radius:6px;}
.risk-high{color:#D55E00;font-weight:700;}.risk-low{color:#009E73;font-weight:700;}
.help-note{font-size:13px;line-height:1.7;color:var(--muted);margin-top:10px;}
.global-image{max-width:100%;height:auto;display:block;margin:auto;}
.footer{font-size:12px;line-height:1.65;color:var(--muted);border-top:1px solid #dfe5eb;margin-top:20px;padding-top:12px;}
@media(max-width:1000px){.app-wrap{padding:15px;}.app-grid{grid-template-columns:minmax(245px,33%) minmax(0,1fr);gap:12px;}.app-title{font-size:23px;}}
@media(max-width:720px){.app-grid{grid-template-columns:1fr;}.risk-grid{grid-template-columns:1fr;}.app-title{font-size:21px;}}
"
ui <- fluidPage(theme=bs_theme(version = 5, primary = "#1F4E78", secondary = "#6C757D", bg = "#FFFFFF", fg = "#243746"),
  tags$head(tags$meta(name="viewport",content="width=device-width, initial-scale=1"),tags$style(HTML(css))),
  div(class="app-wrap",
    div(class="header-line",div(class="app-title","学龄前儿童超重/肥胖风险预测网页计算器"),
        div(class="en-title","Preschool Overweight/Obesity Risk Prediction Calculator"),
        div(class="sub-title","XGBoost-Based Interpretable Machine Learning Model"),
        div(class="sub-title","Final model: Extreme Gradient Boosting (XGBoost)")),
    div(class="intro","本计算器基于经内部验证和时间外部验证的XGBoost模型，根据9项儿童及家庭特征估计学龄前儿童超重/肥胖预测风险，并采用TreeSHAP解释各预测因子对个体模型输出的贡献。"),
    div(class="app-grid",
      div(class="input-panel",h3("预测因子 / Predictor inputs"),
          lapply(seq_along(input_ids),input_control),
          div(class="btn-row",actionButton("predict_btn","开始预测",class="btn-primary"),
              actionButton("reset_btn","重置")),
          uiOutput("download_ui"),div(class="help-note","请填写全部9项信息。修改输入后需重新点击开始预测。")),
      div(uiOutput("status_ui"),
        navset_card_tab(id="tabs",selected="risk",
          nav_panel("Risk Prediction",value="risk",uiOutput("risk_ui"),plotOutput("risk_bar",height="115px"),
                    h5("本次输入 / Prediction inputs"),tableOutput("input_table")),
          nav_panel("SHAP Waterfall",value="waterfall",plotOutput("waterfall",height="480px"),
                    div(class="help-note",shap_note),DTOutput("contribution_table")),
          nav_panel("SHAP Force Plot",value="force",plotOutput("force",height="470px"),div(class="help-note",shap_note)),
          nav_panel("Global SHAP Importance",value="importance",uiOutput("importance_ui"),
                    div(class="help-note","基于484例独立外部验证样本的Mean |SHAP|，不随当前输入改变。")),
          nav_panel("Global SHAP Summary",value="summary",uiOutput("beeswarm_ui"),
                    div(class="help-note","外部验证集的全局TreeSHAP分布；分类变量颜色代表编码类别，不代表连续剂量。")),
          nav_panel("About the Model",value="about",uiOutput("about_ui"))))),
    div(class="footer",
      div("免责声明：本工具基于研究数据建立和验证，仅用于科研及辅助风险评估。预测结果不能替代儿童生长发育评价、BMI-for-age评估、专业临床诊断或医疗建议。"),
      div("Disclaimer: This calculator was developed and validated for research and auxiliary risk assessment purposes. Its predictions should not replace professional growth assessment, BMI-for-age evaluation, clinical diagnosis, or medical advice."))))

server <- function(input, output, session) {
  # 所有单例输入、结果仅驻留当前会话内存；不写服务器文件、日志或数据库。
  result <- reactiveVal(NULL); last_error <- reactiveVal(NULL)
  current_values <- reactive({
    out <- lapply(input_ids,function(id) input[[id]])
    names(out) <- cfg$predictor_vars;out
  })
  # 立即比较当前输入与预测快照。修改任何输入后，旧概率、图和下载全部隐藏。
  active_result <- reactive({
    z <- result()
    if(is.null(z) || !identical(z$values,current_values())) return(NULL)
    z
  })
  run_prediction <- function() {
    result(NULL); last_error(NULL)
    vals <- isolate(current_values())
    tryCatch({
      # 先显示可理解的字段校验错误，再调用模型；日志不打印输入。
      tryCatch(make_web_matrix(vals,cfg),error=function(e){last_error(conditionMessage(e));stop("INPUT_INVALID")})
      z <- predict_one(vals,engine);result(z)
      if(!z$shap_ok) showNotification(z$error,type="warning",duration=NULL)
    },error=function(e){
      if(conditionMessage(e)!="INPUT_INVALID") {
        message("Prediction error: ",conditionMessage(e))
        last_error("预测失败，请检查输入值或模型配置。")
      }
    })
  }
  observeEvent(input$predict_btn,run_prediction(),ignoreInit=TRUE)
  observeEvent(input$reset_btn,{
    result(NULL);last_error(NULL)
    for(j in seq_along(input_ids)) {
      v<-cfg$predictor_vars[j]; s<-cfg$schema[[v]]
      if(s$type=="continuous") updateNumericInput(session,input_ids[j],value=as.numeric(cfg$defaults[[v]]))
      else updateSelectInput(session,input_ids[j],selected=as.character(cfg$defaults[[v]]))
    }
    bslib::nav_select("tabs","risk",session=session)
  },ignoreInit=TRUE)
  # 截图模式只在显式?demo=1时执行默认的合成输入；不是真实病例。
  observeEvent(input$tabs,{
    q <- parseQueryString(session$clientData$url_search)
    if(identical(q$demo,"1")) {
      run_prediction()
      if(identical(q$view,"waterfall")) bslib::nav_select("tabs","waterfall",session=session)
    }
  },once=TRUE,ignoreNULL=TRUE)
  output$status_ui <- renderUI({
    q<-parseQueryString(session$clientData$url_search)
    tagList(if(identical(q$demo,"1")) div(class="alert alert-info","论文界面演示：当前为合成输入，非真实儿童病例。"),
      if(!is.null(last_error())) div(class="alert alert-danger",last_error()),
      if(!is.null(result()) && is.null(active_result())) div(class="alert alert-warning","输入已更改，请重新开始预测；上一结果已隐藏。"),
      if(!is.null(active_result()) && !active_result()$shap_ok) div(class="alert alert-warning",active_result()$error))
  })
  output$risk_ui <- renderUI({
    z<-active_result()
    if(is.null(z)) return(div(class="help-note","填写左侧信息并点击“开始预测”，查看风险概率及个体解释。"))
    tagList(div(class="risk-grid",
       div(class="metric-block",div(class="muted","超重/肥胖预测概率"),
           div(class="probability",scales::percent(z$probability,accuracy=.1))),
       div(class="metric-block",div(class="muted","按锁定阈值分类"),
           h4(class=if(z$predicted_class==1) "risk-high" else "risk-low",
              if(z$predicted_class==1) "Higher predicted risk" else "Lower predicted risk"),
           div(sprintf("Locked cutoff = %.6f (%.2f%%)",z$cutoff,100*z$cutoff)))),
       div(class="help-note","风险分组仅用于本模型结果展示，不能据此诊断超重或肥胖。"))
  })
  output$risk_bar <- renderPlot({
    z<-active_result();req(z)
    ggplot()+annotate("segment",x=0,xend=1,y=0,yend=0,linewidth=9,color="#e4ebef")+
      annotate("segment",x=0,xend=z$probability,y=0,yend=0,linewidth=9,color="#0072B2")+
      geom_vline(xintercept=z$cutoff,linetype=2,color="#D55E00",linewidth=.8)+
      annotate("point",x=z$probability,y=0,size=4,color="#1F4E78")+
      scale_x_continuous(limits=c(0,1),labels=scales::label_percent(),breaks=seq(0,1,.25))+
      labs(x=sprintf("Orange dashed line: locked cutoff %.2f%%",100*z$cutoff),y=NULL)+
      theme_minimal(base_size=11)+theme(axis.text.y=element_blank(),panel.grid=element_blank(),axis.title.x=element_text(color="#6C757D"))
  },res=110)
  output$input_table <- renderTable({
    z<-active_result();req(z);d<-clinical_display(z$values,cfg)
    data.frame(Variable=cfg$predictor_vars,Value=vapply(d,function(x) as.character(x[1]),character(1)),check.names=FALSE)
  },striped=TRUE,bordered=FALSE,spacing="xs")
  shap_result <- reactive({z<-active_result();req(z);shiny::validate(shiny::need(z$shap_ok,z$error));z})
  output$waterfall <- renderPlot({
    z<-shap_result()
    tryCatch({print(shapviz::sv_waterfall(z$sv,row_id=1,max_display=9,fill_colors=c("#D55E00","#0072B2"))+
      labs(title="Individual SHAP Waterfall",subtitle=sprintf("Predicted probability = %.1f%% | Baseline = %.3f | Final margin = %.3f",100*z$probability,z$bias,z$margin),
           x="SHAP contribution / model output (log-odds)")+plot_theme)},
      error=function(e){message("Waterfall rendering: ",conditionMessage(e));shiny::validate(shiny::need(FALSE,"瀑布图暂不可用；请查看下方实际贡献表。"))})
  },res=120)
  fallback_plot <- function(z) {
    d<-data.frame(Feature=unname(cfg$feature_label_map[names(z$shap)]),SHAP=unname(z$shap))
    d$Feature<-factor(d$Feature,levels=d$Feature[order(abs(d$SHAP))])
    ggplot(d,aes(SHAP,Feature,fill=SHAP>=0))+geom_col(width=.65)+geom_vline(xintercept=0,color="#6C757D")+
      scale_fill_manual(values=c("FALSE"="#0072B2","TRUE"="#D55E00"),guide="none")+
      labs(title="Individual SHAP Contributions",subtitle=sprintf("Force plot fallback | Predicted probability = %.1f%%",100*z$probability),x="SHAP value (log-odds)",y=NULL)+plot_theme
  }
  output$force <- renderPlot({
    z<-shap_result()
    # print也在tryCatch内，兼容构图成功但图形设备渲染失败的情况。
    tryCatch({
      assert("sv_force" %in% getNamespaceExports("shapviz"),"此shapviz版本不支持sv_force。")
      print(shapviz::sv_force(z$sv,row_id=1,max_display=9,fill_colors=c("#D55E00","#0072B2"))+
        labs(title="Individual SHAP Force Plot",subtitle=sprintf("Predicted probability = %.1f%% | Baseline %.3f -> margin %.3f",100*z$probability,z$bias,z$margin),x="Model output (log-odds)")+plot_theme)
    },error=function(e){message("Force fallback: ",conditionMessage(e));print(fallback_plot(z))})
  },res=120)
  contribution_data <- reactive({
    z<-shap_result();d<-clinical_display(z$values,cfg)
    data.frame(Feature=cfg$predictor_vars,Input_value=vapply(d,function(x) as.character(x[1]),character(1)),
      SHAP=unname(z$shap),Direction=ifelse(z$shap>0,"Increase predicted risk",ifelse(z$shap<0,"Decrease predicted risk","No contribution")),
      Absolute_SHAP=abs(unname(z$shap)),check.names=FALSE) %>% arrange(desc(Absolute_SHAP))
  })
  output$contribution_table <- renderDT({
    DT::datatable(contribution_data(),rownames=FALSE,options=list(dom="t",paging=FALSE,scrollX=TRUE)) %>%
      DT::formatRound(c("SHAP","Absolute_SHAP"),digits=4)
  })
  output$download_ui <- renderUI({if(!is.null(active_result())) downloadButton("download","下载本次结果（CSV）")})
  output$download <- downloadHandler(filename=function() paste0("XGBoost_prediction_",format(Sys.time(),"%Y%m%d_%H%M%S"),".csv"),
    content=function(file){
      z<-active_result();req(z);d<-clinical_display(z$values,cfg)
      row<-data.frame(Model="Final locked XGBoost",Timestamp=z$timestamp,
        Predicted_probability=z$probability,Locked_cutoff=z$cutoff,Predicted_class=z$predicted_class,
        Risk_group=if(z$predicted_class==1) "Higher predicted risk" else "Lower predicted risk",
        SHAP_scale="Model margin / log-odds",SHAP_QC=if(z$shap_ok) "Passed" else "Unavailable",check.names=FALSE)
      for(v in cfg$predictor_vars) row[[paste0("Input_",v)]]<-as.character(d[[v]][1])
      for(v in cfg$predictor_vars) row[[paste0("SHAP_",v)]]<-if(z$shap_ok) z$shap[[v]] else NA_real_
      row$Baseline_margin<-if(z$shap_ok) z$bias else NA_real_
      row$Predicted_margin<-if(z$shap_ok) z$margin else NA_real_
      con<-file(file,open="wb");on.exit(close(con));writeBin(as.raw(c(239,187,191)),con)
      writeBin(charToRaw(enc2utf8(paste(capture.output(write.csv(row,row.names=FALSE,na="")),collapse="\r\n"))),con)
    },contentType="text/csv")
  output$importance_ui <- renderUI({
    if(!is.null(engine$global)) plotOutput("global_importance",height="530px")
    else if(file.exists("www/XGBoost_SHAP_feature_importance.png"))
      tags$img(src="XGBoost_SHAP_feature_importance.png",class="global-image",alt="External validation global SHAP importance")
    else div(class="help-note","全局SHAP重要性摘要/图片尚未提供；个体预测仍可正常使用。")
  })
  output$global_importance <- renderPlot({
    req(engine$global);d<-engine$global$global_importance
    d$Feature<-factor(unname(cfg$feature_label_map[d$Variable]),levels=rev(unname(cfg$feature_label_map[d$Variable])))
    ggplot(d,aes(Mean_abs_SHAP,Feature))+geom_col(width=.68,fill="#0072B2")+
      labs(x="Mean absolute SHAP value (log-odds)",y=NULL,title="Global SHAP Importance",subtitle="Independent external validation cohort (n = 484)")+plot_theme
  },res=120)
  output$beeswarm_ui <- renderUI({
    if(file.exists("www/XGBoost_SHAP_beeswarm.png")) tags$img(src="XGBoost_SHAP_beeswarm.png",class="global-image",alt="External validation TreeSHAP beeswarm")
    else div(class="help-note","正式外部验证SHAP beeswarm图片尚未提供；个体预测仍可正常使用。")
  })
  output$about_ui <- renderUI({
    perf<-cfg$model_information$performance
    tagList(h4("Final locked XGBoost"),p("开发集：1547例；内部验证集：387例；独立时间外部验证集：484例。"),
      p("固定9项预测因子；网页仅调用已锁定模型和训练阶段预处理参数。"),
      p(sprintf("分类阈值：%.6f（训练阶段确定；p ≥ cutoff归入较高预测风险组）。",cfg$locked_cutoff)),
      p("解释方法：XGBoost原生TreeSHAP；进食速度的dummy贡献按个体有符号相加，还原为9项临床变量。"),
      if(is.data.frame(perf) && nrow(perf)>0) {
        tags$table(class="table table-sm",tags$thead(tags$tr(lapply(names(perf),tags$th))),
                   tags$tbody(lapply(seq_len(nrow(perf)),function(i) tags$tr(lapply(perf[i,],function(x) tags$td(as.character(x)))))))
      } else p("Detailed model performance is reported in the thesis."),
      p(shap_note),p("输入发送至服务器用于当前会话计算；本程序不主动持久化输入记录，不收集姓名、编号、联系方式等身份信息。"))
  })
}
shinyApp(ui,server)
