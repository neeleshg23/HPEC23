LoadTrace_ROOT="/data/pengmiao/ML-DPC-50/LoadTraces"

for app1 in ${app_list[*]}; do
	app="${app1::-7}"
	dvc exp run -n vit-${app} -S apps.app=${app1} --queue
	dvc exp run -n mlp-${app} -S teacher.models.0=m -S teacher.models.1=m -S teacher.models.2=m -S teacher.models.3=m -S teacher.models.4=m -S student.model=m apps.app=${app1} --queue
	dvc exp run -n resnet-${app} -S teacher.models.0=r -S teacher.models.1=r -S teacher.models.2=r -S teacher.models.3=r -S teacher.models.4=r -S student.model=r apps.app=${app1} --queue
	dvc exp run -n densenet-${app} -S teacher.models.0=d -S teacher.models.1=d -S teacher.models.2=d -S teacher.models.3=d -S teacher.models.4=d -S student.model=d apps.app=${app1} --queue
	dvc exp run -n lstm-${app} -S teacher.models.0=l -S teacher.models.1=l -S teacher.models.2=l -S teacher.models.3=l -S teacher.models.4=l -S student.model=l apps.app=${app1} --queue
done
