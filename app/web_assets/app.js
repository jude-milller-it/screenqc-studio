'use strict';
const $=id=>document.getElementById(id);
const token=new URLSearchParams(location.search).get('token')||'';
const strings={
 zh:{labBase:'01 · DeepLabV3+ 基础训练版',labFull:'02 · DeepLabV3+ 完整优化版',segBase:'03 · SegFormer 最终训练权重',segFinal:'04 · SegFormer 最终推理优化',fdsFinal:'05 · FDSNet 推理校正候选',ablScale:'尺度适配',ablRoi:'尺度+ROI',ablConf:'尺度+置信度',ablMix:'尺度+ROI+置信度',ablBad:'历史负优化：形态学+面积限制',ablFds:'FDSNet 原作者原始推理',indLab:'DeepLabV3+ · 完整优化版',indFds:'FDSNet · 推理校正候选',navwork:'工作区',navindustrial:'工业检测',navresearch:'论文实验',navsettings:'设置与环境',localonly:'本地运行 · 数据不上传',researchtool:'研究 / 工业检测工作台',ready:'就绪',workmode:'MODE 01 / INSPECTION',ititle:'工业图像检测',isub:'从图片选择、分割定位到批量报告，一条流程完成。',pixelnote:'像素级缺陷候选 · 人工复核',inputs:'选择图片',includefolder:'支持单张、多张、整个目录',addimages:'添加图片',addfolder:'添加文件夹',clear:'清空',noimages:'暂无待检测图片',addhint:'点击添加图片，选择完成后即展示预览',iparams:'检测配置',modelselect:'检测模型',calibrationnote:'批量输出各类预测像素、候选区域与原图坐标；支持按批次复核。',advanced:'高级选项',minarea:'最小导出区域(px)',deeplabvariant:'DeepLab权重方案',runinspection:'开始检测并生成报告',imagepreview:'图像预览',previewhint:'选择一张图片查看内容',previewmissing:'尚未选中图片',iresults:'检测结果与报告',idetail:'完成后自动导出区域坐标、叠加图、掩码以及图文结合的完整批次报告。',norun:'暂无完成的任务',researchmode:'MODE 02 / RESEARCH',rtitle:'论文实验与证据报告',rsub:'统一评测 → 分类别证据 → 图表与原生Word表格 → PDF / HTML / MD',dataset:'评测数据集',editsettings:'在设置中修改 ↗',split:'评测划分',splitwarning:'评测数据集：val（240张）',variants:'论文对比方案',threearch:'三种架构 · 五项主要对照',cnncomparison:'卷积网络对照（方案3）',labBaseDesc:'独立baseline权重 · 没有则取消选中',labFullDesc:'独立完整训练权重 · CNN对照',segBaseDesc:'512×512 · 原始推理，无额外规则',segFinalDesc:'同一权重 · 尺度+ROI+置信度+油污优化',fdsFinalDesc:'原作者权重 · 污渍概率校正（探索配置）',uiscale:'界面缩放',commonrules:'同权重 · 高分辨率+ROI及类别阈值',finalstage:'同权重 · 再加油污决策优化',externalbaseline:'公开权重 · 外部强基线',stages:'展开SegFormer中间推理阶段（消融）',stagewarning:'这两项对照的是推理阶段；若要比较训练策略，必须有对应的不同训练权重。',reportmethod:'生成方式',standardreport:'标准证据报告',standarddesc:'本地计算 · 无需API',aireport:'AI辅助深度分析',aidesc:'先计算真实指标，再按章节嵌入AI解读',aiconfig:'厂商、模型ID、API地址和Key统一在“设置与环境”配置。',runresearch:'运行统一评测并生成完整报告',sizelink:'微小缺陷尺寸分布',auditlink:'检查数据划分',historical:'从现有结果生成报告',reportoutput:'论文报告输出',reportdesc:'报告按“研究问题—实验设计—结果—各类缺陷—消融—局限”编排。图片与文字就近呈现，完整原始数据放附录。',pdpview:'PDF / HTML 预览',previewreport:'完成评测后可直接在这里预览报告',openfull:'新窗口打开 ↗',settingsmode:'WORKSPACE / PREFERENCES',stitle:'设置与运行环境',ssub:'语言、模型路径、数据目录、AI服务与迁移集中管理。',general:'常规设置',language:'界面语言',datasetroot:'MSD数据集目录',outputroot:'默认结果保存目录',browse:'浏览',savesettings:'保存设置',modelsandweights:'模型与本地权重',oldproject:'旧项目目录',migrate:'从旧项目复制权重',doctor:'环境检查',aiintegration:'AI 报告服务（可选）',ainostore:'Windows加密保存API Key；再次启动自动使用',provider:'服务商',modelid:'模型ID',apiurl:'API地址',aisafety:'API仅接收结构化汇总数值，默认不发送原图、标签或权重。AI解读需要人工校验。',details:'详情',stop:'停止',tasklog:'任务日志',done:'完成',failed:'失败',running:'运行中',pdf:'PDF报告',word:'Word可编辑',html:'HTML可复制',md:'Markdown',json:'结构化数据',openfolder:'打开输出目录',images:'张图片',selected:'待检测图片',nodata:'尚无检测记录',preview:'预览',needkey:'请先在设置里填入API Key',saved:'设置已保存',defaultdata:'当前使用本地MSD目录',splitempty:'为空，不能作为独立测试集',nostats:'没有可用的评测结果',aiagain:'对现有结果生成AI报告',reports:'报告文件',olddata:'使用已有评测结果',copied:'已复制',source:'数据源',missing:'缺少',available:'可用',settingsfirst:'请先配置数据集和模型权重',runfinished:'操作完成',choosing:'正在打开系统文件选择器…',cancel:'取消任务',errors:'执行失败，请查看任务日志',reanalysis:'重新生成报告',selOpt:'SegFormer · 全部推理优化',selBase:'SegFormer · 推理改进',selRaw:'SegFormer · 原始推理512',selScale:'SegFormer · 尺度适配',scheme3:'3 · 完整版',scheme0:'0 · 基线',scheme1:'1 · 损失优化',scheme2:'2 · CBAM',rawName:'SegFormer · 原始推理',rawNote:'512×512 · Argmax · 无ROI/阈值',baseName:'SegFormer · 推理改进',optName:'SegFormer · 油污优化版',scaleName:'SegFormer · 仅尺度适配',scaleNote:'640×1152 · Argmax · 同一训练权重'},
 en:{labBase:'01 · DeepLabV3+ baseline',labFull:'02 · DeepLabV3+ fully optimized',segBase:'03 · SegFormer final trained checkpoint',segFinal:'04 · SegFormer full inference optimization',fdsFinal:'05 · FDSNet calibrated candidate',ablScale:'Resolution only',ablRoi:'Scale + ROI',ablConf:'Scale + confidence gating',ablMix:'Scale + ROI + confidence',ablBad:'Negative ablation: morphology + max-area',ablFds:'FDSNet original inference',indLab:'DeepLabV3+ · Fully optimized',indFds:'FDSNet · Calibrated candidate',navwork:'WORKSPACE',navindustrial:'Inspection',navresearch:'Research',navsettings:'Settings',localonly:'Local-first · no image uploads',researchtool:'Research / inspection workspace',ready:'Ready',workmode:'MODE 01 / INSPECTION',ititle:'Visual Inspection',isub:'From image selection to segmentation, location export and a complete batch report.',pixelnote:'Pixel-level candidates · human review',inputs:'Select images',includefolder:'Single files / multiple files / folders',addimages:'Add images',addfolder:'Add folder',clear:'Clear',noimages:'No images selected',addhint:'Choose images to see a preview',iparams:'Inspection setup',modelselect:'Inference model',calibrationnote:'Exports class-level predicted pixels, candidate regions and original-image coordinates.',advanced:'Advanced options',minarea:'Min exported area (px)',deeplabvariant:'DeepLab checkpoint scheme',runinspection:'Run inspection & build report',imagepreview:'Image preview',previewhint:'Select an image to preview',previewmissing:'No image selected',iresults:'Outputs & report',idetail:'On completion, export coordinates, overlays, masks and a structured batch report.',norun:'No completed run',researchmode:'MODE 02 / RESEARCH',rtitle:'Research Benchmark',rsub:'Evaluation → class-level evidence → editable tables & charts → PDF / HTML / Markdown',dataset:'Dataset',editsettings:'Edit in settings ↗',split:'Dataset split',splitwarning:'Evaluation split: validation set',variants:'Benchmark variants',threearch:'Three architectures · five core configurations',cnncomparison:'CNN comparator (scheme 3)',labBaseDesc:'Distinct baseline checkpoint required',labFullDesc:'Distinct fully trained checkpoint',segBaseDesc:'512×512 · unmodified inference',segFinalDesc:'Same checkpoint · scale+ROI+threshold+oil filter',fdsFinalDesc:'Author checkpoint · exploratory stain calibration',uiscale:'UI scale',commonrules:'Same checkpoint · high resolution + ROI and class thresholds',finalstage:'Shared checkpoint · plus oil decision rule',externalbaseline:'Public checkpoint · external strong baseline',stages:'Show intermediate SegFormer inference stages (ablation)',stagewarning:'These compare inference settings with the SAME final checkpoint; training ablation needs distinct historical weights.',reportmethod:'Report mode',standardreport:'Evidence-first report',standarddesc:'Local calculations · no API',aireport:'AI-assisted interpretation',aidesc:'Compute metrics first; integrate comments next to their charts',aiconfig:'Provider, model ID, endpoint and API key are configured in Settings.',runresearch:'Evaluate & build complete report',sizelink:'Analyze defect sizes',auditlink:'Audit dataset splits',historical:'Build from existing results',reportoutput:'Research deliverables',reportdesc:'Organized by question, protocol, results, classes, ablations and limitations; full data in appendix.',pdpview:'PDF / HTML preview',previewreport:'Run a benchmark to preview the report here',openfull:'Open in new tab ↗',settingsmode:'WORKSPACE / PREFERENCES',stitle:'Settings & Environment',ssub:'Manage language, model paths, dataset, AI providers and migration.',general:'General',language:'Interface language',datasetroot:'MSD dataset root',outputroot:'Default output directory',browse:'Browse',savesettings:'Save settings',modelsandweights:'Models & checkpoints',oldproject:'Previous project folder',migrate:'Copy original checkpoints',doctor:'Environment check',aiintegration:'AI report service (optional)',ainostore:'Encrypted key saved per Windows user',provider:'Provider',modelid:'Model ID',apiurl:'API base URL',aisafety:'Only aggregated structured metrics are sent, not raw images, labels or model weights. Review all AI text.',details:'Details',stop:'Stop',tasklog:'Task log',done:'Completed',failed:'Failed',running:'Running',pdf:'PDF report',word:'Editable Word',html:'Copyable HTML',md:'Markdown',json:'Structured JSON',openfolder:'Open output folder',images:'images',selected:'Selected images',nodata:'No inspection data yet',preview:'Preview',needkey:'Enter an API key in Settings first',saved:'Settings saved',defaultdata:'Using local MSD folder',splitempty:'empty — not an independent test set',nostats:'No evaluation results found',aiagain:'AI report from current data',reports:'Report files',olddata:'Existing benchmark data',copied:'Copied',source:'Source',missing:'Missing',available:'Available',settingsfirst:'Configure dataset and checkpoints first',runfinished:'Operation complete',choosing:'Opening system file picker…',cancel:'Cancel task',errors:'Task failed; inspect logs',reanalysis:'Regenerate report',selOpt:'SegFormer · Full inference optimization',selBase:'SegFormer · Inference tuned',selRaw:'SegFormer · Raw inference 512',selScale:'SegFormer · Scale only',scheme3:'3 · Combined',scheme0:'0 · Baseline',scheme1:'1 · Loss tuned',scheme2:'2 · CBAM',rawName:'SegFormer · Raw inference',rawNote:'512×512 · Argmax · No ROI/threshold',baseName:'SegFormer · Inference tuned',optName:'SegFormer · Oil optimized',scaleName:'SegFormer · Scale only',scaleNote:'640×1152 · Argmax · Same trained checkpoint'}
};
Object.assign(strings.zh,{
 batchGuideTitle:'批次报告与图片',batchGuideText:'在下方选择历史批次，即可查看标准报告、AI报告和全部检测图片。',selectBatch:'选择一个批次查看详情',
 historyTitle:'历史检测批次',historyHelp:'选择批次后查看报告、筛选和排序图片',refreshBatches:'刷新批次',loadingHistory:'正在读取批次…',
 filterClass:'缺陷筛选',sort:'排序',allImages:'全部图片',oil:'油污',stain:'污渍',scratch:'划痕',filename:'文件名',sortPixels:'缺陷像素数量 ↓',sortRegions:'连通区域数 ↓',sortTime:'处理时间 ↓',previous:'上一页',next:'下一页',
 saveKey:'保存API密钥',keySaved:'✓ 密钥已安全保存（当前Windows用户）',keyUnsaved:'尚未保存API密钥',
 reportStandard:'标准报告',reportAI:'AI分析报告',reportMissing:'尚未生成',reportCreateAI:'由标准报告生成AI增强版',reportFolder:'打开报告文件夹',batch:'批次',folder:'文件夹',emptyHistory:'暂无检测历史',historyLoadFail:'历史记录读取失败',
 imageCount:'张',pixelsCount:'预测像素',regionCount:'连通区域',timeMs:'处理耗时',pageEmpty:'无匹配图片',taskCenter:'任务中心',tasksCount:'条记录',activeCount:'进行中',collapse:'收起',expand:'展开',hideTask:'隐藏任务',log:'日志',stopTask:'取消',reportOpen:'打开报告',
 taskPredict:'工业检测',taskEvaluate:'论文评测',taskAI:'AI报告',taskAnalyze:'缺陷尺寸分析',taskAudit:'数据审计',taskPaper:'论文材料',taskMigrate:'权重迁移',taskDoctor:'环境检查',stateRunning:'执行中',stateQueued:'等待中',stateDone:'已完成',stateFailed:'失败',stateCancelled:'已取消',
 latestReport:'最近生成的报告',researchAssets:'研究材料',researchAssetsHelp:'可编辑Word表格与图表、混淆矩阵均单独保存。',
 guardTitle:'有效显示区域过滤（可选）',guardOff:'关闭 · 保留完整盖板结果',guardOn:'开启 · 限定显示区域（实验性）',guardHelp:'显示区域约束仅供额外实验，可能遗漏真实边缘划痕。',inset:'显示区域边缘内缩（像素）',
 ablScaleDesc:'640×1152 · 无额外规则',ablRoiDesc:'仅增加显示区域限制',ablConfDesc:'仅增加类别阈值',ablMixDesc:'油污+0.30之前',ablBadDesc:'展示既有负优化结果',ablEdge:'边缘误检专项 · 最终SegFormer + ROI',ablEdgeDesc:'额外边缘内缩16px，可能遗漏真实划痕',ablFdsDesc:'核验1.10系数的收益',weightFull:'DeepLabV3+ · 完整优化权重',weightBase:'DeepLabV3+ · 基础训练权重'
});
Object.assign(strings.en,{
 batchGuideTitle:'Batch reports and inspection history',batchGuideText:'Select a batch below to access standard/AI reports and inspect all images.',selectBatch:'Select a batch to view details',
 historyTitle:'Inspection history',historyHelp:'Select a batch, view reports, filter and sort inspection images',refreshBatches:'Refresh batches',loadingHistory:'Loading history…',
 filterClass:'Filter by defect',sort:'Sort',allImages:'All images',oil:'Oil',stain:'Stain',scratch:'Scratch',filename:'Filename',sortPixels:'Predicted pixels ↓',sortRegions:'Regions ↓',sortTime:'Processing time ↓',previous:'Previous',next:'Next',
 saveKey:'Save API key',keySaved:'✓ API key securely saved for this Windows user',keyUnsaved:'No API key saved',
 reportStandard:'Standard report',reportAI:'AI-enhanced report',reportMissing:'Not generated yet',reportCreateAI:'Generate AI report from standard report',reportFolder:'Open report folder',batch:'Batch',folder:'Folder',emptyHistory:'No inspection batches found',historyLoadFail:'Failed to load history',
 imageCount:'images',pixelsCount:'Predicted pixels',regionCount:'Regions',timeMs:'Processing time',pageEmpty:'No matching images',taskCenter:'Task center',tasksCount:'records',activeCount:'active',collapse:'Collapse',expand:'Expand',hideTask:'Hide task',log:'Log',stopTask:'Cancel',reportOpen:'Open report',
 taskPredict:'Inspection',taskEvaluate:'Benchmark',taskAI:'AI report',taskAnalyze:'Defect-size analysis',taskAudit:'Dataset audit',taskPaper:'Research materials',taskMigrate:'Checkpoint migration',taskDoctor:'Environment check',stateRunning:'Running',stateQueued:'Queued',stateDone:'Completed',stateFailed:'Failed',stateCancelled:'Cancelled',
 latestReport:'Latest report',researchAssets:'Research assets',researchAssetsHelp:'Editable Word tables, figures and confusion matrices are stored separately.',
 guardTitle:'Display-area filter (optional)',guardOff:'Off · preserve the full surface',guardOn:'On · display area only (experimental)',guardHelp:'The filter is experimental and may miss genuine edge scratches.',inset:'Display-area inward margin (px)',
 ablScaleDesc:'640×1152 · no extra rules',ablRoiDesc:'Add ROI restriction only',ablConfDesc:'Add per-class thresholds only',ablMixDesc:'Before oil +0.30',ablBadDesc:'Historical degradation control',ablEdge:'Edge-error experiment · final SegFormer + ROI',ablEdgeDesc:'16px inward mask may suppress real scratches',ablFdsDesc:'Check the 1.10 calibration effect',weightFull:'DeepLabV3+ · optimized checkpoint',weightBase:'DeepLabV3+ · baseline checkpoint'
});
let app={lang:'zh',page:'inspection',files:[],folders:[],picked:null,dataset:'',lastTask:0,lastPhase:'',runs:{paper:'',inspection:''},report:'',settingsReady:false,rendered:{paper:null,inspection:null},jobsSeen:new Set(),taskFinishedAt:{},tasksExpanded:false,hiddenTasks:new Set(),autoAI:{},historyRun:'',historyOffset:0,historyTotal:0,selectedLog:0};
function t(k){return strings[app.lang][k]||k}
function notify(message,ms=3500){const e=$('toast');e.textContent=message;e.hidden=false;clearTimeout(e._timer);e._timer=setTimeout(()=>e.hidden=true,ms)}
async function api(url,data){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-Session-Token':token},body:JSON.stringify(data)});const v=await r.json();if(!r.ok)throw Error(v.error||'Operation failed');return v}
async function get(url){const r=await fetch(url);if(!r.ok)throw Error('HTTP '+r.status);return r.json()}
function basename(p){return String(p).split(/[\\/]/).pop()}
function translate(){if(app.settingsReady&&$('aiKeySaved'))$('aiKeySaved').textContent=t(app.aiKeySaved?'keySaved':'keyUnsaved');document.documentElement.lang=app.lang==='zh'?'zh-CN':'en';document.querySelectorAll('[data-i18n]').forEach(el=>{const key=el.getAttribute('data-i18n');el.textContent=t(key)});$('langZh').classList.toggle('selected',app.lang==='zh');$('langEn').classList.toggle('selected',app.lang==='en');$('settingLanguage').value=app.lang;updateNavLabel();updateFiles();updateResults(true);}
function updateNavLabel(){$('breadcrumb').textContent='SCREENQC / '+({inspection:'INSPECTION',research:'RESEARCH',settings:'SETTINGS'}[app.page]||'')}
function navigate(pg){app.page=pg;document.querySelectorAll('.page').forEach(el=>el.classList.toggle('current',el.id===pg));document.querySelectorAll('.nav-item').forEach(el=>el.classList.toggle('active',el.dataset.target===pg));updateNavLabel();}
function previews(path){app.picked=path;}
function updateFiles(){const div=$('fileList');div.replaceChildren();const total=app.files.length+app.folders.length;if(!total){div.innerHTML=`<div class="empty"><div class="empty-ico">▧</div><strong>${t('noimages')}</strong><span>${t('addhint')}</span></div>`;previews(null);return}let paths=[...app.folders,...app.files];for(const path of paths.slice(0,200)){const button=document.createElement('button');button.className='file-item'+(app.picked===path?' selected':'');button.title=path;button.textContent=(app.folders.includes(path)?'▤ ':'▧ ')+basename(path);button.onclick=()=>{if(app.files.includes(path))previews(path);else{app.picked=path}updateFilesSelection()};div.append(button)}if(total>200){const p=document.createElement('div');p.className='muted-note';p.textContent=`+${total-200}...`;div.append(p)}if(!app.picked&&app.files.length)previews(app.files[0]);}
function updateFilesSelection(){document.querySelectorAll('.file-item').forEach(x=>x.classList.toggle('selected',x.title===app.picked))}
async function choose(kind,initial){notify(t('choosing'),8000);return (await api('/api/choose',{kind,initial})).selection}
async function addFiles(){try{const picks=await choose('files',app.dataset+'/JPEGImages');for(const file of picks){if(!app.files.includes(file))app.files.push(file)}updateFiles();notify(`${app.files.length} ${t('images')}`)}catch(ex){notify(ex.message)}}
async function addFolder(){try{const pick=await choose('folder',app.dataset+'/JPEGImages');if(pick&&!app.folders.includes(pick))app.folders.push(pick);updateFiles();notify(`${app.folders.length} folder(s)`)}catch(ex){notify(ex.message)}}
function canRun(){return true}
async function start(action,payload={}){try{if(!canRun())throw Error(t('running'));const status=await api('/api/start',{action,...payload});handleStatus(status);return (status.tasks||[])[0]?.id||true}catch(ex){notify(ex.message,5500);return false}}
function selectedModels(){return [...document.querySelectorAll('.model-option input:checked')].map(x=>x.value)}
async function runResearch(){let models=selectedModels();if(!models.length){notify(app.lang==='zh'?'请选择至少一个模型':'Select at least one model');return}const mode=document.querySelector('input[name="researchReport"]:checked').value;if(mode==='ai'&&!($('aiKey').value.trim()||app.aiKeySaved)){navigate('settings');notify(t('needkey'));return}const success=await start('evaluate',{dataset:app.dataset,split:$('evalSplit').value,models,samples:2,deeplab_scheme:3});if(success){if(mode==='ai')app.autoAI[success]='paper';navigate('research')}}
async function runInspection(){if(!app.files.length&&!app.folders.length){notify(t('noimages'));return}let ai=document.querySelector('input[name="inspectionReport"]:checked').value==='ai';if(ai&&!($('aiKey').value.trim()||app.aiKeySaved)){navigate('settings');notify(t('needkey'));return}const success=await start('predict',{files:app.files,folders:app.folders,model:$('inspectionModel').value,deeplab_scheme:3,min_area:Number($('minArea').value)});if(success&&ai)app.autoAI[success]='industrial'}
async function runAI(mode,source){try{if(!($('aiKey').value.trim()||app.aiKeySaved)){navigate('settings');throw Error(t('needkey'))}let res=await api('/api/ai',{mode,source,provider:$('aiProvider').value,model:$('aiModel').value,base:$('aiBase').value,key:$('aiKey').value});handleStatus(res)}catch(e){notify(e.message)}}
function link(path,label){const a=document.createElement('a');a.className='result-action';a.href='/api/result?path='+encodeURIComponent(path);a.target='_blank';a.rel='noopener';a.textContent=label+' ↗';return a}
function setReportFrame(report){if(!report)return;app.report=report;const stage=$('researchPdf');stage.replaceChildren();let iframe=document.createElement('iframe');iframe.src='/api/result?path='+encodeURIComponent(report)+'#toolbar=1';iframe.title='PDF report';stage.append(iframe);$('previewSwitch').hidden=false}
async function openFolder(path){try{await api('/api/open-folder',{path})}catch(e){notify(e.message)}}
function el(tag,cls,txt){const e=document.createElement(tag);if(cls)e.className=cls;if(txt!==undefined)e.textContent=txt;return e}
function folderButton(path){const b=el('button','button neutral',t('reportFolder'));b.onclick=()=>openFolder(path);return b}
async function drawReportHub(container,source,mode){
 container.classList.remove('result-empty');container.classList.add('report-hub');
 container.replaceChildren();
 if(!source){container.append(el('p','muted-note',t('norun')));return}
 const q=new URLSearchParams({run:source,mode});let data;
 try{data=await get('/api/report-catalog?'+q.toString())}catch(ex){container.append(el('p','muted-note',ex.message));return}
 if(!container.isConnected)return;
 const row=el('div','report-double');container.append(row);
 for(const kind of ['standard','ai']){
   const section=el('section','report-kind');row.append(section);
   section.append(el('h3','',t(kind==='standard'?'reportStandard':'reportAI')));
   const files=data[kind]?.files||{}, formats=['PDF','Word','HTML','Markdown'];
   if(!Object.keys(files).length)section.append(el('p','muted-note',t('reportMissing')));
   const buttons=el('div','report-links');section.append(buttons);
   for(const format of formats)if(files[format])buttons.append(link(files[format],format));
   if(data[kind]?.directory)buttons.append(folderButton(data[kind].directory));
 }
 const generate=el('button','button neutral ai-generate',t('reportCreateAI'));
 generate.disabled=!data.standard||!Object.keys(data.standard.files||{}).length;
 generate.onclick=()=>runAI(mode==='paper'?'paper':'industrial',source);
 container.append(generate);
}
function resultPanel(element,source,mode){const container=$(element);if(!container||!source)return;drawReportHub(container,source,mode)}
function updateResults(force=false){const source=app.runs.paper;if(force||app.rendered.paper!==source){resultPanel('researchResults',source,'paper');app.rendered.paper=source}}
function renderSplits(counts){let wrap=$('splitChips');wrap.replaceChildren();let labels=['train','val','test','trainval'];for(const key of labels){let span=document.createElement('span');span.className='split-chip'+(counts[key]===0?' emptysplit':'');span.textContent=key+' · '+(counts[key]??'—');wrap.append(span)}$('settingsSplitHint').textContent=Object.entries(counts).map(([k,v])=>k+': '+(v??'—')).join(' · ');let selected=$('evalSplit').value;for(const option of $('evalSplit').options)option.disabled=!counts[option.value];if(!counts[selected])$('evalSplit').value=counts.val?'val':labels.find(k=>counts[k])||'val'}
function renderWeights(weights){const el=$('weightChecks');el.replaceChildren();for(const [k,yes] of Object.entries(weights)){let item=document.createElement('div');item.className='check-item';item.innerHTML='<strong>'+k+'</strong><span class="check-dot '+(yes?'':'warn')+'">'+(yes?'● '+t('available'):'● '+t('missing'))+'</span>';el.append(item)}}
async function refresh(){try{const v=await get('/api/state');if(!app.settingsReady){app.lang=v.settings.language||'zh';app.dataset=v.settings.dataset||'';$('settingDataset').value=app.dataset;$('settingOutput').value=v.settings.outputs||'';for(const mod of ['deeplab','segformer','fdsnet']){let key='weight'+mod[0].toUpperCase()+mod.slice(1);$(key).value=v.weights_paths?.[mod]||'';}$('weightDeeplabBase').value=v.weights_paths?.deeplab_base||'';
 $('settingScale').value=String(v.settings.ui_scale||110);
 applyScale(Number(v.settings.ui_scale||110));
 let ai=v.settings.ai||{};
 $('aiProvider').value=ai.provider||'DeepSeek';$('aiModel').value=ai.model||'deepseek-flash';$('aiBase').value=ai.base||'https://api.deepseek.com';
 app.aiKeySaved=Boolean(v.ai_key_saved);
 $('aiKeySaved').textContent=t(app.aiKeySaved?'keySaved':'keyUnsaved');
 $('oldProject').value=v.settings.old_project;app.settingsReady=true;translate()}else{app.dataset=v.settings.dataset||app.dataset} app.aiKeySaved=Boolean(v.ai_key_saved)||app.aiKeySaved; $('datasetPath').textContent=app.dataset;renderSplits(v.split_counts);renderWeights(v.weights);app.runs={...v.last};if(!v.task||v.task.phase==='idle')updateResults();handleStatus(v)}catch(e){$('connection').textContent='● Offline'}}
function handleStatus(v){
 if(v.last)app.runs={...app.runs,...v.last};
 const jobs=v.tasks||[];const dock=$('taskDock');dock.replaceChildren();
 const active=jobs.filter(t=>t.phase==='running'||t.phase==='queued');
 const recent=jobs.filter(t=>!['running','queued'].includes(t.phase)&&app.taskFinishedAt[t.id]&&Date.now()-app.taskFinishedAt[t.id]<12000);
 const display=app.tasksExpanded?jobs.filter(t=>!app.hiddenTasks.has(t.id)).slice(0,12):[];
 if(jobs.length){const toggle=document.createElement('button');toggle.className='dock-toggle';toggle.textContent=`${t('taskCenter')} · ${active.length} ${t('activeCount')} · ${jobs.length} ${t('tasksCount')} ${app.tasksExpanded?'− '+t('collapse'):'＋ '+t('expand')}`;toggle.onclick=()=>{app.tasksExpanded=!app.tasksExpanded;handleStatus(v)};dock.append(toggle)}
 for(const task of display){
  const card=document.createElement('div');card.className='dock-job '+task.phase;
  const head=document.createElement('div');head.className='dock-head';
  const title=document.createElement('strong');title.textContent=(t('task'+({predict:'Predict',evaluate:'Evaluate',ai:'AI',analyze:'Analyze',audit:'Audit',paper:'Paper',migrate:'Migrate',doctor:'Doctor'}[task.action]||'Paper')))+' · '+t('state'+({running:'Running',queued:'Queued',done:'Done',failed:'Failed',cancelled:'Cancelled'}[task.phase]||'Done'));
  head.append(title);
  const hide=document.createElement('button');hide.textContent='×';hide.title=t('hideTask');hide.onclick=()=>{app.hiddenTasks.add(task.id);handleStatus(v)};head.append(hide);
  const log=document.createElement('button');log.textContent=t('log');log.onclick=()=>{$('logText').textContent=(task.lines||[]).join('\n');$('logModal').hidden=false};head.append(log);
  if(task.phase==='running'||task.phase==='queued'){const cancel=document.createElement('button');cancel.textContent=t('stopTask');cancel.onclick=()=>api('/api/cancel',{id:task.id}).catch(e=>notify(e.message));head.append(cancel)}
  card.append(head);
  const line=document.createElement('small');line.textContent=task.error||task.progress||(task.lines||[]).slice(-1)[0]||'';card.append(line);
  if(task.phase==='done'&&task.report){const a=link(task.report,t('reportOpen'));card.append(a)}
  dock.append(card);
 }
 for(const task of jobs){
  if(!['done','failed','cancelled'].includes(task.phase)||app.jobsSeen.has(task.id))continue;
  app.jobsSeen.add(task.id);app.taskFinishedAt[task.id]=Date.now();
  if(task.phase!=='done'){notify(task.error||'任务执行失败',5500);continue}
  if(task.action==='predict')app.runs.inspection=task.result;
  if(task.action==='evaluate'||task.action==='paper')app.runs.paper=task.result;
  if(task.action==='ai'){
   if(task.report.includes('工业'))app.runs.inspection=task.result;
   else app.runs.paper=task.result;
  }
  if(task.action==='analyze'||task.action==='audit'){
    if(task.report)window.open('/api/result?path='+encodeURIComponent(task.report),'_blank');
  }
  updateResults(true);
  if(task.action==='predict'||(task.action==='ai'&&task.report&&task.report.includes('工业')))refreshHistory();
  const kind=app.autoAI[task.id];if(kind){delete app.autoAI[task.id];runAI(kind,task.result)}
 }
}
function applyScale(value){document.documentElement.style.fontSize=(16*value/100)+'px';document.body.style.zoom=String(value/100)}
async function saveSettings(){try{
 const dataset=$('settingDataset').value.trim();const lang=$('settingLanguage').value;
 const weights={deeplab:$('weightDeeplab').value,segformer:$('weightSegformer').value,fdsnet:$('weightFdsnet').value};
 const ai={provider:$('aiProvider').value,model:$('aiModel').value.trim(),base:$('aiBase').value.trim()};
 await api('/api/settings',{settings:{dataset,language:lang,outputs:$('settingOutput').value.trim(),weights,
   deeplab_base:$('weightDeeplabBase').value.trim(),ui_scale:Number($('settingScale').value),ai},api_key:$('aiKey').value.trim()});
 app.dataset=dataset;app.lang=lang;app.aiKeySaved=app.aiKeySaved||Boolean($('aiKey').value.trim());
 $('aiKey').value='';$('aiKeySaved').textContent=t(app.aiKeySaved?'keySaved':'keyUnsaved');
 applyScale(Number($('settingScale').value));translate();await refresh();notify(t('saved'))
 }catch(ex){notify(ex.message,5000)}}
async function saveApiKey(){
 const key=$('aiKey').value.trim();if(!key){notify(t('needkey'));return}
 const b=$('saveApiKey');b.disabled=true;
 try{const response=await api('/api/credentials',{api_key:key});if(!response.saved)throw Error('Credential verification failed');
  app.aiKeySaved=true;$('aiKey').value='';$('aiKeySaved').textContent=t('keySaved');notify(t('keySaved'));
 }catch(ex){notify(ex.message,6500)}finally{b.disabled=false}
}
function syncProvider(){let p=$('aiProvider').value;let presets={'DeepSeek':['https://api.deepseek.com','deepseek-flash'],'OpenAI':['https://api.openai.com/v1','gpt-4.1-mini'],'硅基流动 SiliconFlow':['https://api.siliconflow.cn/v1',''],'自定义 OpenAI兼容':['','']};if(presets[p]){$('aiBase').value=presets[p][0];$('aiModel').value=presets[p][1]}}
async function refreshHistory(){
 try{
  const data=await get('/api/history?type=inspection');const list=$('historyRuns');list.replaceChildren();
  if(!data.runs.length){list.textContent=t('emptyHistory');$('historyReports').replaceChildren();$('historyGallery').replaceChildren();return}
  if(!data.runs.some(r=>r.path===app.historyRun))app.historyRun=data.runs[0].path;
  for(const run of data.runs){const button=el('button','history-run'+(app.historyRun===run.path?' active':''));
   button.append(el('strong','',run.created));button.append(el('small','',run.images+' '+t('imageCount')+' · '+(run.model||'—')));
   button.onclick=()=>{app.historyRun=run.path;app.historyOffset=0;refreshHistory();};list.append(button);
  }
  await loadHistoryImages();
 }catch(e){notify(t('historyLoadFail')+': '+e.message)}
}
async function loadHistoryImages(){
 if(!app.historyRun)return;
 try{
  const filter=$('historyClass').value,sort=$('historySort').value;
  const q=new URLSearchParams({run:app.historyRun,class:filter,sort,offset:app.historyOffset,limit:24});
  const data=await get('/api/inspection-images?'+q.toString());app.historyTotal=data.total;
  const sel=$('historySelection');sel.replaceChildren();
  sel.append(el('strong','batch-title',t('batch')+' '+basename(app.historyRun)+' · '+data.total+' '+t('imageCount')));
  await drawReportHub($('historyReports'),app.historyRun,'inspection');
  const gallery=$('historyGallery');gallery.replaceChildren();
  for(const item of data.items){const card=el('a','history-image');card.href='/api/result?path='+encodeURIComponent(item.image);card.target='_blank';card.rel='noopener';
   const img=document.createElement('img');img.loading='lazy';img.src=card.href;img.alt=item.filename;card.append(img);
   card.append(el('strong','',item.filename));
   const pixels=Object.values(item.pixels).reduce((a,b)=>a+b,0).toLocaleString();
   card.append(el('small','',`${t('pixelsCount')} ${pixels} · ${t('regionCount')} ${item.regions} · ${item.time_ms}ms`));gallery.append(card)
  }
  $('historyPage').textContent=data.total?`${app.historyOffset+1}–${Math.min(app.historyOffset+24,data.total)} / ${data.total}`:t('pageEmpty');
  $('historyPrev').disabled=app.historyOffset===0;$('historyNext').disabled=app.historyOffset+24>=data.total;
 }catch(e){notify(e.message)}
}
function setup(){document.querySelectorAll('.nav-item').forEach(b=>b.onclick=()=>navigate(b.dataset.target));$('langZh').onclick=()=>{app.lang='zh';$('settingLanguage').value='zh';translate();updateResults(true);refreshHistory();api('/api/settings',{settings:{language:'zh'}}).catch(()=>{})};$('langEn').onclick=()=>{app.lang='en';$('settingLanguage').value='en';translate();updateResults(true);refreshHistory();api('/api/settings',{settings:{language:'en'}}).catch(()=>{})};$('settingLanguage').onchange=()=>{app.lang=$('settingLanguage').value;translate()};$('settingScale').onchange=()=>applyScale(Number($('settingScale').value));
$('addImages').onclick=addFiles;$('addFolder').onclick=addFolder;$('clearImages').onclick=()=>{app.files=[];app.folders=[];app.picked=null;updateFiles()};$('runInspection').onclick=runInspection;$('runResearch').onclick=runResearch;$('editDataset').onclick=()=>navigate('settings');$('saveSettings').onclick=saveSettings;$('saveApiKey').onclick=saveApiKey;$('browseDataset').onclick=async()=>{try{let p=await choose('folder',$('settingDataset').value);if(p)$('settingDataset').value=p}catch(e){notify(e.message)}};$('browseOutput').onclick=async()=>{try{let p=await choose('folder',$('settingOutput').value);if(p)$('settingOutput').value=p}catch(e){notify(e.message)}};document.querySelectorAll('.weightBrowse').forEach(btn=>{btn.onclick=async()=>{try{const mod=btn.dataset.weight;let key=mod==='deeplab_base'?'weightDeeplabBase':'weight'+mod[0].toUpperCase()+mod.slice(1);let p=await choose('file',$(key).value);if(p)$(key).value=p}catch(e){notify(e.message)}}});$('browseOld').onclick=async()=>{try{let p=await choose('folder',$('oldProject').value);if(p)$('oldProject').value=p}catch(e){notify(e.message)}};$('doMigrate').onclick=()=>start('migrate',{old_project:$('oldProject').value});$('doDoctor').onclick=()=>start('doctor');$('runSize').onclick=()=>start('analyze',{dataset:app.dataset,split:$('evalSplit').value});$('runAudit').onclick=()=>start('audit',{dataset:app.dataset});$('makeFromExisting').onclick=()=>start('paper',{source:app.runs.paper||''});$('aiProvider').onchange=syncProvider;$('showTaskLog').onclick=()=>{$('logModal').hidden=false};$('closeModal').onclick=()=>{$('logModal').hidden=true};$('logModal').onclick=e=>{if(e.target===$('logModal'))$('logModal').hidden=true};$('cancelTask').onclick=async()=>{if(confirm(t('cancel')))await api('/api/cancel',{})};$('historyRefresh').onclick=refreshHistory;$('historyClass').onchange=()=>{app.historyOffset=0;loadHistoryImages()};$('historySort').onchange=()=>{app.historyOffset=0;loadHistoryImages()};$('historyPrev').onclick=()=>{app.historyOffset=Math.max(0,app.historyOffset-24);loadHistoryImages()};$('historyNext').onclick=()=>{app.historyOffset+=24;loadHistoryImages()};}
setup();refresh();refreshHistory();setInterval(refresh,1800);
